#include "atomicsolver.hpp"
#include <integratorxx/quadratures/all.hpp>
#include <cassert>
#include <xc.h>
#include <cmath>
#include <openorbitaloptimizer/scfsolver.hpp>
#include <nlohmann/json.hpp>
#include "cmdline.h"

namespace OpenOrbitalOptimizer {
  // Instantiate all types of SCFSolver just to check it compiles
  template class SCFSolver<double, false>;

  namespace AtomicSolver {

    std::pair<bool, bool> eval_xc(const Eigen::MatrixXd & rho, const Eigen::MatrixXd & sigma, const Eigen::MatrixXd & tau, Eigen::VectorXd & exc, Eigen::MatrixXd & vxc, Eigen::MatrixXd & vsigma, Eigen::MatrixXd & vtau, int func_id, int nspin) {
      // Initialize functional
      xc_func_type func;
      if(xc_func_init(&func, func_id, nspin) != 0) {
        std::ostringstream oss;
        oss << "Functional "<<func_id<<" not found!";
        throw std::runtime_error(oss.str());
      }

      // Energy density and potentials
      Eigen::Index N = rho.rows();
      exc.setZero(N);
      if(nspin==1) {
        vxc.setZero(1, N);
        vsigma.setZero(1, N);
        vtau.setZero(1, N);
      } else {
        vxc.setZero(2, N);
        vsigma.setZero(3, N);
        vtau.setZero(2, N);
      }

      // Transpose rho, sigma, and tau so the spin index runs fastest in
      // memory, matching libxc's interleaved (up, dn, up, dn, ...) layout.
      Eigen::MatrixXd rhot   = rho.transpose();
      Eigen::MatrixXd sigmat = sigma.transpose();
      Eigen::MatrixXd taut   = tau.transpose();

      bool gga=false, mgga=false;
      double *lapl = nullptr, *vlapl = nullptr;
      switch(func.info->family)
        {
        case XC_FAMILY_LDA:
          xc_lda_exc_vxc(&func, N, rhot.data(), exc.data(), vxc.data());
          break;
        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
          xc_gga_exc_vxc(&func, N, rhot.data(), sigmat.data(), exc.data(), vxc.data(), vsigma.data());
          gga=true;
          break;
        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
        case XC_FAMILY_HYB_MGGA:
#endif
          xc_mgga_exc_vxc(&func, N, rhot.data(), sigmat.data(), lapl, taut.data(), exc.data(), vxc.data(), vsigma.data(), vlapl, vtau.data());
          gga=true;
          mgga=true;
          break;

        default:
          throw std::logic_error("Case not handled!\n");
        }

      // Back-transpose to (N, nspin*) layout
      vxc    = vxc.transpose().eval();
      vsigma = vsigma.transpose().eval();
      vtau   = vtau.transpose().eval();

      return std::make_pair(gga,mgga);
    }

    std::tuple<double,std::vector<Eigen::MatrixXd>> build_xc_unpolarized(const std::vector<std::shared_ptr<const RadialBasis>> & basis, const std::vector<Eigen::MatrixXd> & P, size_t N, int func_id) {
      assert(basis.size() == P.size());

      // Handle case of no functional
      if(func_id==-1) {
        std::vector<Eigen::MatrixXd> F(basis.size());
        for(size_t l=0;l<basis.size();l++)
          F[l] = Eigen::MatrixXd::Zero(basis[l]->nbf(),basis[l]->nbf());
        return std::make_tuple(0.0,F);
      }

      // Get radial grid
      IntegratorXX::TreutlerAhlrichs<double,double> quad(N);
      const auto & quad_w = quad.weights();
      const auto & quad_r = quad.points();
      Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(quad_w.data(), quad_w.size());
      Eigen::VectorXd r = Eigen::Map<const Eigen::VectorXd>(quad_r.data(), quad_r.size());
      // Angular factor
      double angfac = 4.0*M_PI;

      // Evaluate basis functions
      std::vector<Eigen::MatrixXd> bf(basis.size()), df(basis.size());
      for(size_t l=0;l<basis.size();l++) {
        bf[l]=basis[l]->eval_f(r);
        df[l]=basis[l]->eval_df(r);
      }

      // Electron density (single column => store as Nx1 matrix to match
      // the libxc convention used by eval_xc above)
      Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(r.size(),1);
      for(size_t l=0;l<basis.size();l++)
        rho.col(0) += (bf[l]*P[l]/angfac*bf[l].transpose()).diagonal();

      // Density gradient
      Eigen::VectorXd drho = Eigen::VectorXd::Zero(r.size());
      for(size_t l=0;l<basis.size();l++)
        drho += 2.0*(df[l]*P[l]/angfac*bf[l].transpose()).diagonal();
      // Reduced gradient
      Eigen::MatrixXd sigma = drho.array().square().matrix();

      // Kinetic energy density
      Eigen::MatrixXd tau = Eigen::MatrixXd::Zero(r.size(),1);
      for(size_t l=0;l<basis.size();l++)
        tau.col(0) += 0.5*(df[l]*P[l]/angfac*df[l].transpose()).diagonal();
      for(size_t l=1;l<basis.size();l++)
        tau.col(0).array() += 0.5*l*(l+1)*(bf[l]*P[l]/angfac*bf[l].transpose()).diagonal().array() / r.array().square();

      // Energy density and potentials
      Eigen::VectorXd exc;
      Eigen::MatrixXd vxc;
      Eigen::MatrixXd vsigma;
      Eigen::MatrixXd vtau;
      auto ggamgga = eval_xc(rho, sigma, tau, exc, vxc, vsigma, vtau, func_id, XC_UNPOLARIZED);
      bool gga = std::get<0>(ggamgga);
      bool mgga = std::get<1>(ggamgga);

      // xc energy
      double E = angfac*(exc.array()*rho.col(0).array()).matrix().dot((w.array()*r.array().square()).matrix());

      // Fock matrix, LDA term
      Eigen::VectorXd r2 = r.array().square().matrix();
      Eigen::VectorXd w_r2 = (w.array()*r2.array()).matrix();
      std::vector<Eigen::MatrixXd> F(basis.size());
      for(size_t l=0;l<basis.size();l++) {
        Eigen::VectorXd d = (w_r2.array()*vxc.col(0).array()).matrix();
        F[l] = bf[l].transpose() * d.asDiagonal() * bf[l];
      }
      if(gga) {
        for(size_t l=0;l<basis.size();l++) {
          Eigen::VectorXd d = (w_r2.array() * vsigma.col(0).array() * drho.array()).matrix();
          Eigen::MatrixXd Fgga = 2*df[l].transpose() * d.asDiagonal() * bf[l];
          F[l] += Fgga + Fgga.transpose();
        }
      }
      if(mgga) {
        for(size_t l=0;l<basis.size();l++) {
          Eigen::VectorXd d = (w_r2.array() * vtau.col(0).array()).matrix();
          F[l] += 0.5*df[l].transpose() * d.asDiagonal() * df[l];
          if(l>0) {
            Eigen::VectorXd d2 = (w.array() * vtau.col(0).array()).matrix();
            F[l] += 0.5*l*(l+1)*bf[l].transpose() * d2.asDiagonal() * bf[l];
          }
        }
      }

      return std::make_tuple(E,F);
    }

    std::tuple<double,std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> build_xc_polarized(const std::vector<std::shared_ptr<const RadialBasis>> & basis, const std::vector<Eigen::MatrixXd> & Pa, const std::vector<Eigen::MatrixXd> & Pb, size_t N, int func_id) {
      assert(basis.size() == Pa.size());
      assert(basis.size() == Pb.size());

      // Handle case of no functional
      if(func_id==-1) {
        std::vector<Eigen::MatrixXd> Fa(basis.size()), Fb(basis.size());
        for(size_t l=0;l<basis.size();l++) {
          Fa[l] = Eigen::MatrixXd::Zero(basis[l]->nbf(),basis[l]->nbf());
          Fb[l] = Eigen::MatrixXd::Zero(basis[l]->nbf(),basis[l]->nbf());
        }
        return std::make_tuple(0.0,Fa,Fb);
      }

      // Get radial grid
      IntegratorXX::TreutlerAhlrichs<double,double> quad(N);
      const auto & quad_w = quad.weights();
      const auto & quad_r = quad.points();
      Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(quad_w.data(), quad_w.size());
      Eigen::VectorXd r = Eigen::Map<const Eigen::VectorXd>(quad_r.data(), quad_r.size());
      double angfac = 4.0*M_PI;

      // Evaluate basis functions
      std::vector<Eigen::MatrixXd> bf(basis.size()), df(basis.size());
      for(size_t l=0;l<basis.size();l++) {
        bf[l]=basis[l]->eval_f(r);
        df[l]=basis[l]->eval_df(r);
      }

      // Electron density: column 0 = alpha, column 1 = beta
      Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(r.size(),2);
      for(size_t l=0;l<basis.size();l++) {
        rho.col(0) += (bf[l]*Pa[l]/angfac*bf[l].transpose()).diagonal();
        rho.col(1) += (bf[l]*Pb[l]/angfac*bf[l].transpose()).diagonal();
      }
      Eigen::VectorXd rhotot = rho.col(0) + rho.col(1);

      // Density gradient
      Eigen::MatrixXd drho = Eigen::MatrixXd::Zero(r.size(),2);
      for(size_t l=0;l<basis.size();l++) {
        drho.col(0) += 2.0*(df[l]*Pa[l]/angfac*bf[l].transpose()).diagonal();
        drho.col(1) += 2.0*(df[l]*Pb[l]/angfac*bf[l].transpose()).diagonal();
      }
      // Reduced gradient
      Eigen::MatrixXd sigma(r.size(),3);
      sigma.col(0) = (drho.col(0).array()*drho.col(0).array()).matrix();
      sigma.col(1) = (drho.col(0).array()*drho.col(1).array()).matrix();
      sigma.col(2) = (drho.col(1).array()*drho.col(1).array()).matrix();

      // Kinetic energy density
      Eigen::MatrixXd tau = Eigen::MatrixXd::Zero(r.size(),2);
      for(size_t l=0;l<basis.size();l++) {
        tau.col(0) += 0.5*(df[l]*Pa[l]/angfac*df[l].transpose()).diagonal();
        tau.col(1) += 0.5*(df[l]*Pb[l]/angfac*df[l].transpose()).diagonal();
      }
      for(size_t l=1;l<basis.size();l++) {
        tau.col(0).array() += 0.5*l*(l+1)*(bf[l]*Pa[l]/angfac*bf[l].transpose()).diagonal().array() / r.array().square();
        tau.col(1).array() += 0.5*l*(l+1)*(bf[l]*Pb[l]/angfac*bf[l].transpose()).diagonal().array() / r.array().square();
      }

      // Energy density and potentials
      Eigen::VectorXd exc;
      Eigen::MatrixXd vxc;
      Eigen::MatrixXd vsigma;
      Eigen::MatrixXd vtau;
      auto ggamgga = eval_xc(rho, sigma, tau, exc, vxc, vsigma, vtau, func_id, XC_POLARIZED);
      bool gga = std::get<0>(ggamgga);
      bool mgga = std::get<1>(ggamgga);

      // xc energy
      double E = angfac*(exc.array()*rhotot.array()).matrix().dot((w.array()*r.array().square()).matrix());

      Eigen::VectorXd r2   = r.array().square().matrix();
      Eigen::VectorXd w_r2 = (w.array()*r2.array()).matrix();

      // Fock matrix, LDA term
      std::vector<Eigen::MatrixXd> Fa(basis.size()), Fb(basis.size());
      for(size_t l=0;l<basis.size();l++) {
        Eigen::VectorXd da = (w_r2.array()*vxc.col(0).array()).matrix();
        Eigen::VectorXd db = (w_r2.array()*vxc.col(1).array()).matrix();
        Fa[l] = bf[l].transpose() * da.asDiagonal() * bf[l];
        Fb[l] = bf[l].transpose() * db.asDiagonal() * bf[l];
      }
      if(gga) {
        for(size_t l=0;l<basis.size();l++) {
          Eigen::VectorXd da = (w_r2.array()*(2*vsigma.col(0).array()*drho.col(0).array() + vsigma.col(1).array()*drho.col(1).array())).matrix();
          Eigen::MatrixXd Fagga = df[l].transpose() * da.asDiagonal() * bf[l];
          Fa[l] += Fagga + Fagga.transpose();
          Eigen::VectorXd db = (w_r2.array()*(2*vsigma.col(2).array()*drho.col(1).array() + vsigma.col(1).array()*drho.col(0).array())).matrix();
          Eigen::MatrixXd Fbgga = df[l].transpose() * db.asDiagonal() * bf[l];
          Fb[l] += Fbgga + Fbgga.transpose();
        }
      }
      if(mgga) {
        for(size_t l=0;l<basis.size();l++) {
          Eigen::VectorXd da = (w_r2.array()*vtau.col(0).array()).matrix();
          Eigen::VectorXd db = (w_r2.array()*vtau.col(1).array()).matrix();
          Fa[l] += 0.5*df[l].transpose() * da.asDiagonal() * df[l];
          Fb[l] += 0.5*df[l].transpose() * db.asDiagonal() * df[l];
          if(l>0) {
            Eigen::VectorXd da2 = (w.array()*vtau.col(0).array()).matrix();
            Eigen::VectorXd db2 = (w.array()*vtau.col(1).array()).matrix();
            Fa[l] += 0.5*l*(l+1)*bf[l].transpose() * da2.asDiagonal() * bf[l];
            Fb[l] += 0.5*l*(l+1)*bf[l].transpose() * db2.asDiagonal() * bf[l];
          }
        }
      }

      return std::make_tuple(E,Fa,Fb);
    }

    std::tuple<double,std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> build_xc_neo(const std::vector<std::shared_ptr<const RadialBasis>> & pbasis, const std::vector<Eigen::MatrixXd> & Pp, const std::vector<std::shared_ptr<const RadialBasis>> & ebasis, const std::vector<Eigen::MatrixXd> & Pe, size_t N, int func_id) {
      assert(pbasis.size() == Pp.size());
      assert(ebasis.size() == Pe.size());

      // Get radial grid
      IntegratorXX::TreutlerAhlrichs<double,double> quad(N);
      const auto & quad_w = quad.weights();
      const auto & quad_r = quad.points();
      Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(quad_w.data(), quad_w.size());
      Eigen::VectorXd r = Eigen::Map<const Eigen::VectorXd>(quad_r.data(), quad_r.size());
      double angfac = 4.0*M_PI;

      // Evaluate protonic and electronic basis functions
      std::vector<Eigen::MatrixXd> pbf(pbasis.size()), pdf(pbasis.size());
      for(size_t l=0;l<pbasis.size();l++) {
        pbf[l]=pbasis[l]->eval_f(r);
        pdf[l]=pbasis[l]->eval_df(r);
      }
      std::vector<Eigen::MatrixXd> ebf(ebasis.size()), edf(ebasis.size());
      for(size_t l=0;l<ebasis.size();l++) {
        ebf[l]=ebasis[l]->eval_f(r);
        edf[l]=ebasis[l]->eval_df(r);
      }

      // Proton and electron densities
      Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(r.size(),2);
      for(size_t l=0;l<pbasis.size();l++) {
        rho.col(0) += (pbf[l]*Pp[l]/angfac*pbf[l].transpose()).diagonal();
      }
      for(size_t l=0;l<ebasis.size();l++) {
        rho.col(1) += (ebf[l]*Pe[l]/angfac*ebf[l].transpose()).diagonal();
      }
      Eigen::VectorXd rhotot = rho.col(0)+rho.col(1);

      // Density gradient
      Eigen::MatrixXd drho = Eigen::MatrixXd::Zero(r.size(),2);
      for(size_t l=0;l<pbasis.size();l++) {
        drho.col(0) += 2.0*(pdf[l]*Pp[l]/angfac*pbf[l].transpose()).diagonal();
      }
      for(size_t l=0;l<ebasis.size();l++) {
        drho.col(1) += 2.0*(edf[l]*Pe[l]/angfac*ebf[l].transpose()).diagonal();
      }
      // Reduced gradient
      Eigen::MatrixXd sigma(r.size(),3);
      sigma.col(0) = (drho.col(0).array()*drho.col(0).array()).matrix();
      sigma.col(1) = (drho.col(0).array()*drho.col(1).array()).matrix();
      sigma.col(2) = (drho.col(1).array()*drho.col(1).array()).matrix();

      // Kinetic energy density
      Eigen::MatrixXd tau = Eigen::MatrixXd::Zero(r.size(),2);
      for(size_t l=0;l<pbasis.size();l++) {
        tau.col(0) += 0.5*(pdf[l]*Pp[l]/angfac*pdf[l].transpose()).diagonal();
        if(l>0)
          tau.col(0).array() += 0.5*l*(l+1)*(pbf[l]*Pp[l]/angfac*pbf[l].transpose()).diagonal().array()/r.array().square();
      }
      for(size_t l=0;l<ebasis.size();l++) {
        tau.col(1) += 0.5*(edf[l]*Pe[l]/angfac*edf[l].transpose()).diagonal();
        if(l>0)
          tau.col(1).array() += 0.5*l*(l+1)*(ebf[l]*Pe[l]/angfac*ebf[l].transpose()).diagonal().array()/r.array().square();
      }

      // Energy density and potentials
      Eigen::VectorXd exc;
      Eigen::MatrixXd vxc;
      Eigen::MatrixXd vsigma;
      Eigen::MatrixXd vtau;
      auto ggamgga = eval_xc(rho, sigma, tau, exc, vxc, vsigma, vtau, func_id, XC_POLARIZED);
      bool gga = std::get<0>(ggamgga);
      bool mgga = std::get<1>(ggamgga);

      // xc energy
      double E = angfac*(exc.array()*rhotot.array()).matrix().dot((w.array()*r.array().square()).matrix());

      Eigen::VectorXd r2   = r.array().square().matrix();
      Eigen::VectorXd w_r2 = (w.array()*r2.array()).matrix();

      // Fock matrices, LDA term
      std::vector<Eigen::MatrixXd> Fp(pbasis.size()), Fe(ebasis.size());
      for(size_t l=0;l<pbasis.size();l++) {
        Eigen::VectorXd d = (w_r2.array()*vxc.col(0).array()).matrix();
        Fp[l] = pbf[l].transpose() * d.asDiagonal() * pbf[l];
      }
      for(size_t l=0;l<ebasis.size();l++) {
        Eigen::VectorXd d = (w_r2.array()*vxc.col(1).array()).matrix();
        Fe[l] = ebf[l].transpose() * d.asDiagonal() * ebf[l];
      }
      if(gga) {
        for(size_t l=0;l<pbasis.size();l++) {
          Eigen::VectorXd d = (w_r2.array()*(2*vsigma.col(0).array()*drho.col(0).array() + vsigma.col(1).array()*drho.col(1).array())).matrix();
          Eigen::MatrixXd Fpgga = pdf[l].transpose() * d.asDiagonal() * pbf[l];
          Fp[l] += Fpgga + Fpgga.transpose();
        }
        for(size_t l=0;l<ebasis.size();l++) {
          Eigen::VectorXd d = (w_r2.array()*(2*vsigma.col(2).array()*drho.col(1).array() + vsigma.col(1).array()*drho.col(0).array())).matrix();
          Eigen::MatrixXd Fegga = edf[l].transpose() * d.asDiagonal() * ebf[l];
          Fe[l] += Fegga + Fegga.transpose();
        }
      }
      if(mgga) {
        for(size_t l=0;l<pbasis.size();l++) {
          Eigen::VectorXd d = (w_r2.array()*vtau.col(0).array()).matrix();
          Fp[l] += 0.5*pdf[l].transpose() * d.asDiagonal() * pdf[l];
          if(l>0) {
            Eigen::VectorXd d2 = (w.array()*vtau.col(0).array()).matrix();
            Fp[l] += 0.5*l*(l+1)*pbf[l].transpose() * d2.asDiagonal() * pbf[l];
          }
        }
        for(size_t l=0;l<ebasis.size();l++) {
          Eigen::VectorXd d = (w_r2.array()*vtau.col(1).array()).matrix();
          Fe[l] += 0.5*edf[l].transpose() * d.asDiagonal() * edf[l];
          if(l>0) {
            Eigen::VectorXd d2 = (w.array()*vtau.col(1).array()).matrix();
            Fe[l] += 0.5*l*(l+1)*ebf[l].transpose() * d2.asDiagonal() * ebf[l];
          }
        }
      }

      return std::make_tuple(E,Fp,Fe);
    }

    std::vector<Eigen::MatrixXd> build_J(const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & basis, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & Pbasis, const std::vector<Eigen::MatrixXd> & P) {
      std::vector<Eigen::MatrixXd> J(basis.size());
      for(size_t i=0;i<basis.size();i++) {
        J[i] = Eigen::MatrixXd::Zero(basis[i]->nbf(),basis[i]->nbf());
      }

      for(size_t lin=0;lin<Pbasis.size();lin++) {
        // Skip zero blocks
        if(P[lin].norm() == 0.0)
          continue;
        for(size_t lout=0;lout<basis.size();lout++) {
          J[lout] += basis[lout]->coulomb(Pbasis[lin],P[lin]);
        }
      }

      return J;
    }

    std::vector<Eigen::MatrixXd> form_X(double linear_dependency_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis) {
      std::vector<Eigen::MatrixXd> X(radial_basis.size());
      for(size_t i=0;i<X.size();i++) {
        // Overlap matrix
        Eigen::MatrixXd S = radial_basis[i]->overlap();

        // Normalization
        Eigen::VectorXd normlz = S.diagonal().array().pow(-0.5).matrix();
        Eigen::MatrixXd Snorm  = normlz.asDiagonal() * S * normlz.asDiagonal();

        // Compute X using canonical orthogonalization
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Snorm);
        Eigen::VectorXd sval = es.eigenvalues();
        Eigen::MatrixXd svec = es.eigenvectors();

        // Find the columns of svec whose eigenvalues clear the threshold
        std::vector<Eigen::Index> sidx;
        sidx.reserve(sval.size());
        for(Eigen::Index k=0;k<sval.size();k++)
          if(sval(k) >= linear_dependency_threshold)
            sidx.push_back(k);

        Eigen::MatrixXd Xi(svec.rows(), sidx.size());
        Eigen::VectorXd scale(sidx.size());
        for(size_t k=0;k<sidx.size();k++) {
          Xi.col(k) = svec.col(sidx[k]);
          scale(k)  = std::pow(sval(sidx[k]), -0.5);
        }
        X[i] = Xi * scale.asDiagonal();
        // Apply normalization
        X[i] = normlz.asDiagonal() * X[i];
      }
      return X;
    }

    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> form_core_hamiltonian(const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis, double Z, double particle_mass = 1.0) {
      std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> Hcore(radial_basis.size());
      for(size_t i=0;i<radial_basis.size();i++) {
        Eigen::MatrixXd T = radial_basis[i]->kinetic(i) / particle_mass;
        Eigen::MatrixXd V = - Z*radial_basis[i]->nuclear_attraction();
        Hcore[i] = std::make_pair(T,V);
      }
      return Hcore;
    }

    double coulomb_energy(const std::vector<Eigen::MatrixXd> & P, const std::vector<Eigen::MatrixXd> & J) {
      assert(P.size()==J.size());
      double E = 0.0;
      for(size_t l=0;l<P.size();l++) {
        E += 0.5*(J[l]*P[l]).trace();
      }
      return E;
    }

    OpenOrbitalOptimizer::SCFSolver<double, false> restricted_scf(int Z, int Q, int x_func_id, int c_func_id, int Ngrid, double linear_dependency_threshold, double convergence_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis, int verbosity, bool core_excitation) {
      // Form the orthogonal orbital basis
      std::vector<Eigen::MatrixXd> X = form_X(linear_dependency_threshold, radial_basis);

      // and the core Hamiltonian
      std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> Hcore = form_core_hamiltonian(radial_basis, Z);

      // Number of blocks per particle type
      OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(1);
      number_of_blocks_per_particle_type(0) = static_cast<Eigen::Index>(radial_basis.size());

      Eigen::VectorXd maximum_occupation(radial_basis.size());
      for(size_t l=0;l<radial_basis.size();l++)
        maximum_occupation(l) = 2*(2*l+1);

      Eigen::VectorXd number_of_particles(1);
      number_of_particles(0) = (double) (Z-Q);

      std::vector<std::string> block_descriptions(radial_basis.size());
      for(size_t l=0;l<radial_basis.size();l++) {
        std::ostringstream oss;
        oss << "l=" << l;
        block_descriptions[l] = oss.str();
      }

      // Form the Fock matrix guess
      OpenOrbitalOptimizer::FockMatrix<double> fock_guess(radial_basis.size());
      for(size_t i=0;i<X.size();i++)
        fock_guess[i] = X[i].transpose() * (Hcore[i].first+Hcore[i].second) * X[i];

      // Fock builder. All inputs / outputs are Eigen-typed now.
      OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [radial_basis, X, Ngrid, x_func_id, c_func_id, Hcore, verbosity](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
        const auto & orbitals = dm.first;
        const auto & occupations = dm.second;

        // Form the density matrix in the original basis
        std::vector<Eigen::MatrixXd> P(orbitals.size());
        for(size_t l=0;l<P.size();l++) {
          // In the orthonormal basis it is
          P[l] = orbitals[l] * occupations[l].asDiagonal() * orbitals[l].transpose();
          // and in the non-orthonormal basis it is
          P[l] = X[l] * P[l] * X[l].transpose();
        }

        // Form the non-orthonormal basis Fock matrix. Build coulomb
        auto coulomb = OpenOrbitalOptimizer::AtomicSolver::build_J(radial_basis, radial_basis, P);
        // Build exchange
        auto exchange = OpenOrbitalOptimizer::AtomicSolver::build_xc_unpolarized(radial_basis, P, Ngrid, x_func_id);
        // and correlation
        auto correlation = OpenOrbitalOptimizer::AtomicSolver::build_xc_unpolarized(radial_basis, P, Ngrid, c_func_id);

        // Collect the Fock matrix in the orthonormal basis
        std::vector<Eigen::MatrixXd> fock(orbitals.size());
        for(size_t l=0; l<fock.size(); l++) {
          fock[l] = X[l].transpose() * (Hcore[l].first + Hcore[l].second + coulomb[l] + std::get<1>(exchange)[l] + std::get<1>(correlation)[l]) * X[l];
        }

        // Calculate energy terms
        double Ekin = 0.0;
        for(size_t l=0;l<P.size();l++)
          Ekin += (Hcore[l].first*P[l]).trace();

        double Enuc = 0.0;
        for(size_t l=0;l<P.size();l++)
          Enuc += (Hcore[l].second*P[l]).trace();

        double Ej = coulomb_energy(P,coulomb);
        double Ex = std::get<0>(exchange);
        double Ec = std::get<0>(correlation);
        double Etot = Ekin+Enuc+Ej+Ex+Ec;

        if(verbosity>=10) {
          printf("Kinetic energy  % .10f\n",Ekin);
          printf("Nuclear energy  % .10f\n",Enuc);
          printf("Coulomb energy  % .10f\n",Ej);
          printf("Exchange energy % .10f\n",Ex);
          printf("Correlation energy % .10f\n",Ec);
          printf("Total energy    % .10f\n",Etot);
        }

        return std::make_pair(Etot, fock);
      };

      // Initialize SCF solver
      OpenOrbitalOptimizer::SCFSolver<double, false> scfsolver(
          number_of_blocks_per_particle_type,
          maximum_occupation,
          number_of_particles,
          fock_builder, block_descriptions);
      scfsolver.verbosity(verbosity);
      scfsolver.convergence_threshold(convergence_threshold);
      scfsolver.initialize_with_fock(fock_guess);
      scfsolver.run();

      if(core_excitation) {
        // Form core-excited state
        auto density_matrix = scfsolver.get_solution();
        auto orbitals =  density_matrix.first;
        auto occupations =  density_matrix.second;
        auto fock_build = scfsolver.get_fock_build();

        // Decrease occupation of 1s orbital
        occupations[0](0) = 0.0;
        scfsolver.frozen_occupations(true);
        scfsolver.initialize_with_orbitals(orbitals, occupations);
        scfsolver.run();
        auto core_hole_fock_build = scfsolver.get_fock_build();
        printf("1s double ionization energy % .3f eV\n",(core_hole_fock_build.first-fock_build.first)*27.2114);
      }

      return scfsolver;
    }

    OpenOrbitalOptimizer::SCFSolver<double, false> unrestricted_scf(int Z, int Q, int M, int x_func_id, int c_func_id, int Ngrid, double linear_dependency_threshold, double convergence_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis, int verbosity, bool core_excitation) {
      // Form the orthogonal orbital basis
      std::vector<Eigen::MatrixXd> X = form_X(linear_dependency_threshold, radial_basis);

      // Form the core Hamiltonian
      std::vector<std::pair<Eigen::MatrixXd,Eigen::MatrixXd>> Hcore = form_core_hamiltonian(radial_basis, Z);

      // Number of blocks per particle type
      OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(2);
      number_of_blocks_per_particle_type(0) = static_cast<Eigen::Index>(radial_basis.size());
      number_of_blocks_per_particle_type(1) = static_cast<Eigen::Index>(radial_basis.size());

      Eigen::VectorXd maximum_occupation(2*radial_basis.size());
      for(size_t l=0;l<radial_basis.size();l++)
        maximum_occupation(l) = 2*l+1;
      for(size_t i=0;i<radial_basis.size();i++) {
        maximum_occupation(i+radial_basis.size()) = maximum_occupation(i);
      }

      bool even_number_of_electrons = ((Z-Q)%2==0);
      bool even_multiplicity = (((M-1)%2)==0);
      if(even_number_of_electrons != even_multiplicity) {
        std::ostringstream oss;
        oss << "Cannot have multiplicity " << M << " with " << Z-Q << " electrons!\n";
        throw std::logic_error(oss.str());
      }

      int Nela = (Z-Q+M-1)/2;
      int Nelb = (Z-Q)-Nela;
      printf("Nela = %i Nelb = %i\n",Nela,Nelb);
      assert(Nela>0);
      assert(Nelb>=0);
      Eigen::VectorXd number_of_particles(2);
      number_of_particles(0) = (double) Nela;
      number_of_particles(1) = (double) Nelb;

      std::vector<std::string> block_descriptions(2*radial_basis.size());
      for(size_t l=0;l<radial_basis.size();l++) {
        std::ostringstream oss;
        oss << "alpha l=" << l;
        block_descriptions[l] = oss.str();
      }
      for(size_t l=0;l<radial_basis.size();l++) {
        std::ostringstream oss;
        oss << "beta l=" << l;
        block_descriptions[l+radial_basis.size()] = oss.str();
      }

      OpenOrbitalOptimizer::FockMatrix<double> fock_guess(2*radial_basis.size());
      for(size_t i=0;i<X.size();i++) {
        fock_guess[i] = X[i].transpose() * (Hcore[i].first+Hcore[i].second) * X[i];
        fock_guess[i+radial_basis.size()] = fock_guess[i];
      }

      // Fock builder
      OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [radial_basis, X, Ngrid, x_func_id, c_func_id, Hcore, verbosity](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
        const auto & orbitals = dm.first;
        const auto & occupations = dm.second;

        // Form the spin-up and spin-down density matrices in the original basis
        assert(orbitals.size()%2==0);
        size_t Nblocks = orbitals.size()/2;
        std::vector<Eigen::MatrixXd> Pa(Nblocks), Pb(Nblocks), Ptot(Nblocks);
        for(size_t iblock=0;iblock<Nblocks;iblock++) {
          size_t ablock = iblock;
          // In the orthonormal basis it is
          Pa[iblock] = orbitals[ablock] * occupations[ablock].asDiagonal() * orbitals[ablock].transpose();
          // and in the non-orthonormal basis it is
          Pa[iblock] = X[iblock] * Pa[iblock] * X[iblock].transpose();
          size_t bblock = iblock+Nblocks;
          // In the orthonormal basis it is
          Pb[iblock] = orbitals[bblock] * occupations[bblock].asDiagonal() * orbitals[bblock].transpose();
          // and in the non-orthonormal basis it is
          Pb[iblock] = X[iblock] * Pb[iblock] * X[iblock].transpose();

          // Since we use same X for both spin channels, total density is
          Ptot[iblock] = Pa[iblock] + Pb[iblock];
        }

        // Form the non-orthonormal basis Fock matrix. Build coulomb
        auto coulomb = OpenOrbitalOptimizer::AtomicSolver::build_J(radial_basis, radial_basis, Ptot);
        // Build exchange
        auto exchange = OpenOrbitalOptimizer::AtomicSolver::build_xc_polarized(radial_basis, Pa, Pb, Ngrid, x_func_id);
        // and correlation
        auto correlation = OpenOrbitalOptimizer::AtomicSolver::build_xc_polarized(radial_basis, Pa, Pb, Ngrid, c_func_id);

        // Collect the Fock matrix in the orthonormal basis
        std::vector<Eigen::MatrixXd> fock(orbitals.size());
        for(size_t iblock=0; iblock<Nblocks; iblock++) {
          // Spin-up Fock
          size_t ablock = iblock, bblock = iblock+Nblocks;
          fock[ablock] = X[iblock].transpose() * (Hcore[iblock].first + Hcore[iblock].second + coulomb[iblock] + std::get<1>(exchange)[iblock] + std::get<1>(correlation)[iblock]) * X[iblock];
          fock[bblock] = X[iblock].transpose() * (Hcore[iblock].first + Hcore[iblock].second + coulomb[iblock] + std::get<2>(exchange)[iblock] + std::get<2>(correlation)[iblock]) * X[iblock];
        }

        // Calculate energy terms
        double Ekin = 0.0;
        for(size_t l=0;l<Ptot.size();l++)
          Ekin += (Hcore[l].first*Ptot[l]).trace();

        double Enuc = 0.0;
        for(size_t l=0;l<Ptot.size();l++)
          Enuc += (Hcore[l].second*Ptot[l]).trace();

        double Ej = coulomb_energy(Ptot, coulomb);
        double Ex = std::get<0>(exchange);
        double Ec = std::get<0>(correlation);
        double Etot = Ekin+Enuc+Ej+Ex+Ec;

        if(verbosity>=10) {
          printf("Kinetic energy     % .10f\n",Ekin);
          printf("Nuclear energy     % .10f\n",Enuc);
          printf("Coulomb energy     % .10f\n",Ej);
          printf("Exchange energy    % .10f\n",Ex);
          printf("Correlation energy % .10f\n",Ec);
          printf("Total energy       % .10f\n",Etot);
        }

        return std::make_pair(Etot, fock);
      };

      // Initialize SCF solver
      OpenOrbitalOptimizer::SCFSolver<double, false> scfsolver(
          number_of_blocks_per_particle_type,
          maximum_occupation,
          number_of_particles,
          fock_builder, block_descriptions);
      scfsolver.verbosity(verbosity);
      scfsolver.convergence_threshold(convergence_threshold);
      scfsolver.initialize_with_fock(fock_guess);
      scfsolver.run();

      if(core_excitation) {
        // Form core-excited state
        auto density_matrix = scfsolver.get_solution();
        auto orbitals =  density_matrix.first;
        auto occupations =  density_matrix.second;
        auto fock_build = scfsolver.get_fock_build();

        // Decrease occupation of 1s orbital
        occupations[0](0) = 0.0;
        scfsolver.frozen_occupations(true);
        scfsolver.initialize_with_orbitals(orbitals, occupations);
        scfsolver.run();
        auto core_hole_fock_build = scfsolver.get_fock_build();
        printf("1s ionization energy % .3f eV\n",(core_hole_fock_build.first-fock_build.first)*27.2114);
      }

      return scfsolver;
    }

    OpenOrbitalOptimizer::SCFSolver<double, false> unrestricted_neo_scf(int Z, int Q, int M, int x_func_id, int c_func_id, int epc_func_id, int Ngrid, double linear_dependency_threshold, double convergence_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & protonic_basis, double proton_mass, int verbosity, bool core_excitation) {
      // Form the orthogonal orbital basis
      std::vector<Eigen::MatrixXd> X  = form_X(linear_dependency_threshold, radial_basis);
      std::vector<Eigen::MatrixXd> Xp = form_X(linear_dependency_threshold, protonic_basis);

      // Form the electronic core Hamiltonian and nuclear kinetic operator
      auto Hcore  = form_core_hamiltonian(radial_basis, Z);
      auto Hpcore = form_core_hamiltonian(protonic_basis, Z, proton_mass);

      // Number of blocks per particle type
      OpenOrbitalOptimizer::IndexVector number_of_blocks_per_particle_type(3);
      number_of_blocks_per_particle_type(0) = static_cast<Eigen::Index>(radial_basis.size());
      number_of_blocks_per_particle_type(1) = static_cast<Eigen::Index>(radial_basis.size());
      number_of_blocks_per_particle_type(2) = static_cast<Eigen::Index>(protonic_basis.size());

      Eigen::VectorXd maximum_occupation(2*radial_basis.size()+protonic_basis.size());
      for(size_t l=0;l<radial_basis.size();l++)
        maximum_occupation(l) = 2*l+1;
      for(size_t l=0;l<radial_basis.size();l++)
        maximum_occupation(l+radial_basis.size()) = 2*l+1;
      for(size_t l=0;l<protonic_basis.size();l++)
        maximum_occupation(l+2*radial_basis.size()) = 2*l+1;

      bool even_number_of_electrons = ((Z-Q)%2==0);
      bool even_multiplicity = (((M-1)%2)==0);
      if(even_number_of_electrons != even_multiplicity) {
        std::ostringstream oss;
        oss << "Cannot have multiplicity " << M << " with " << Z-Q << " electrons!\n";
        throw std::logic_error(oss.str());
      }

      int Nela = (Z-Q+M-1)/2;
      int Nelb = (Z-Q)-Nela;
      printf("Nela = %i Nelb = %i\n",Nela,Nelb);
      assert(Nela>0);
      assert(Nelb>=0);
      Eigen::VectorXd number_of_particles(3);
      number_of_particles(0) = (double) Nela;
      number_of_particles(1) = (double) Nelb;
      number_of_particles(2) = 1.0;

      std::vector<std::string> block_descriptions(2*radial_basis.size()+protonic_basis.size());
      for(size_t l=0;l<radial_basis.size();l++) {
        std::ostringstream oss;
        oss << "alpha l=" << l;
        block_descriptions[l] = oss.str();
      }
      for(size_t l=0;l<radial_basis.size();l++) {
        std::ostringstream oss;
        oss << "beta l=" << l;
        block_descriptions[l+radial_basis.size()] = oss.str();
      }
      for(size_t l=0;l<protonic_basis.size();l++) {
        std::ostringstream oss;
        oss << "proton l=" << l;
        block_descriptions[l+2*radial_basis.size()] = oss.str();
      }

      // Fock builder
      OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [radial_basis, protonic_basis, X, Xp, Ngrid, x_func_id, c_func_id, epc_func_id, Hcore, Hpcore, verbosity](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
        const auto & orbitals = dm.first;
        const auto & occupations = dm.second;

        // Form the spin-up and spin-down density matrices in the original basis
        size_t Nblocks = radial_basis.size();
        std::vector<Eigen::MatrixXd> Pa(Nblocks), Pb(Nblocks), Ptot(Nblocks);
        for(size_t iblock=0;iblock<Nblocks;iblock++) {
          size_t ablock = iblock;
          // In the orthonormal basis it is
          Pa[iblock] = orbitals[ablock] * occupations[ablock].asDiagonal() * orbitals[ablock].transpose();
          // and in the non-orthonormal basis it is
          Pa[iblock] = X[iblock] * Pa[iblock] * X[iblock].transpose();
          size_t bblock = iblock+Nblocks;
          // In the orthonormal basis it is
          Pb[iblock] = orbitals[bblock] * occupations[bblock].asDiagonal() * orbitals[bblock].transpose();
          // and in the non-orthonormal basis it is
          Pb[iblock] = X[iblock] * Pb[iblock] * X[iblock].transpose();

          // Since we use same X for both spin channels, total density is
          Ptot[iblock] = Pa[iblock] + Pb[iblock];
        }
        size_t Npblocks = protonic_basis.size();
        std::vector<Eigen::MatrixXd> Pproton(Npblocks);
        for(size_t iblock=0;iblock<Npblocks;iblock++) {
          size_t pblock = iblock+2*Nblocks;
          Pproton[iblock] = orbitals[pblock] * occupations[pblock].asDiagonal() * orbitals[pblock].transpose();
          Pproton[iblock] = Xp[iblock] * Pproton[iblock] * Xp[iblock].transpose();
        }

        // Form the non-orthonormal basis Fock matrix. Build coulomb
        auto coulomb_ee = OpenOrbitalOptimizer::AtomicSolver::build_J(radial_basis, radial_basis, Ptot);
        auto coulomb_ep = OpenOrbitalOptimizer::AtomicSolver::build_J(radial_basis, protonic_basis, Pproton);
        auto coulomb_pe = OpenOrbitalOptimizer::AtomicSolver::build_J(protonic_basis, radial_basis, Ptot);
        // Build exchange
        auto exchange = OpenOrbitalOptimizer::AtomicSolver::build_xc_polarized(radial_basis, Pa, Pb, Ngrid, x_func_id);
        // and correlation
        auto correlation = OpenOrbitalOptimizer::AtomicSolver::build_xc_polarized(radial_basis, Pa, Pb, Ngrid, c_func_id);
        // and correlation
        auto pcorrelation = OpenOrbitalOptimizer::AtomicSolver::build_xc_neo(protonic_basis, Pproton, radial_basis, Ptot, Ngrid, epc_func_id);

        // Collect the Fock matrix in the orthonormal basis
        std::vector<Eigen::MatrixXd> fock(orbitals.size());
        for(size_t iblock=0; iblock<Nblocks; iblock++) {
          // Spin-up Fock
          size_t ablock = iblock, bblock = iblock+Nblocks;
          fock[ablock] = X[iblock].transpose() * (Hcore[iblock].first + coulomb_ee[iblock] - coulomb_ep[iblock] + std::get<1>(exchange)[iblock] + std::get<1>(correlation)[iblock] + std::get<2>(pcorrelation)[iblock]) * X[iblock];
          fock[bblock] = X[iblock].transpose() * (Hcore[iblock].first + coulomb_ee[iblock] - coulomb_ep[iblock] + std::get<2>(exchange)[iblock] + std::get<2>(correlation)[iblock] + std::get<2>(pcorrelation)[iblock]) * X[iblock];
        }
        for(size_t iblock=0; iblock<Npblocks; iblock++) {
          size_t pblock = iblock+2*Nblocks;
          fock[pblock] = Xp[iblock].transpose() * (Hpcore[iblock].first - coulomb_pe[iblock] + std::get<1>(pcorrelation)[iblock]) * Xp[iblock];
        }

        // Calculate energy terms
        double Ekin = 0.0;
        for(size_t l=0;l<Ptot.size();l++)
          Ekin += (Hcore[l].first*Ptot[l]).trace();

        double Epkin = 0.0;
        for(size_t l=0;l<Pproton.size();l++)
          Epkin += (Hpcore[l].first*Pproton[l]).trace();

        double Ej = coulomb_energy(Ptot,coulomb_ee);
        double Eep = -2*coulomb_energy(Ptot,coulomb_ep);
        double Ex = std::get<0>(exchange);
        double Ec = std::get<0>(correlation);
        double Epc = std::get<0>(pcorrelation);
        double Etot = Ekin+Epkin+Ej+Eep+Ex+Ec+Epc;

        if(verbosity>=10) {
          printf("e kinetic energy       % .10f\n",Ekin);
          printf("p kinetic energy       % .10f\n",Epkin);
          printf("e-e Coulomb energy     % .10f\n",Ej);
          printf("e-p Coulomb energy     % .10f\n",Eep);
          printf("e-e exchange energy    % .10f\n",Ex);
          printf("e-e correlation energy % .10f\n",Ec);
          printf("p-e correlation energy % .10f\n",Epc);
          printf("Total energy           % .10f\n",Etot);
        }

        return std::make_pair(Etot, fock);
      };

      // Initialize SCF solver
      OpenOrbitalOptimizer::SCFSolver<double, false> scfsolver(
          number_of_blocks_per_particle_type,
          maximum_occupation,
          number_of_particles,
          fock_builder, block_descriptions);
      scfsolver.convergence_threshold(convergence_threshold);
      scfsolver.verbosity(verbosity);

      {
        // Run a calculation with the point nucleus to initialize the electronic orbitals
        OpenOrbitalOptimizer::SCFSolver esolver(unrestricted_scf(Z, Q, M, x_func_id, c_func_id, Ngrid, linear_dependency_threshold, convergence_threshold, radial_basis, verbosity, false));
        auto electronic_dm(esolver.get_solution());
        const auto & orbitals = electronic_dm.first;
        const auto & occupations = electronic_dm.second;

        // Compute the electronic density matrix
        size_t Nblocks = radial_basis.size();
        size_t Npblocks = protonic_basis.size();
        std::vector<Eigen::MatrixXd> Ptot(Nblocks);
        for(size_t iblock=0; iblock<Nblocks; iblock++) {
          size_t ablock = iblock;
          size_t bblock = iblock+Nblocks;

          // In the orthonormal basis it is
          Eigen::MatrixXd Pa = X[iblock] * orbitals[ablock] * occupations[ablock].asDiagonal() * orbitals[ablock].transpose() * X[iblock].transpose();
          Eigen::MatrixXd Pb = X[iblock] * orbitals[bblock] * occupations[bblock].asDiagonal() * orbitals[bblock].transpose() * X[iblock].transpose();
          // Since we use same X for both spin channels, total density is
          Ptot[iblock] = Pa + Pb;
        }

        // Guess Fock matrix from converged calculation
        std::vector<Eigen::MatrixXd> fock_guess = esolver.get_fock_build().second;

        // Compute the Coulomb potential
        auto coulomb_pe = OpenOrbitalOptimizer::AtomicSolver::build_J(protonic_basis, radial_basis, Ptot);
        for(size_t iblock=0; iblock<Npblocks; iblock++) {
          fock_guess.push_back(Xp[iblock].transpose() * (Hpcore[iblock].first - coulomb_pe[iblock] ) * Xp[iblock]);
        }

        scfsolver.initialize_with_fock(fock_guess);
      }

      scfsolver.run();

      if(core_excitation) {
        // Form core-excited state
        auto density_matrix = scfsolver.get_solution();
        auto orbitals =  density_matrix.first;
        auto occupations =  density_matrix.second;
        auto fock_build = scfsolver.get_fock_build();

        // Decrease occupation of 1s orbital
        occupations[0](0) = 0.0;
        scfsolver.frozen_occupations(true);
        scfsolver.initialize_with_orbitals(orbitals, occupations);
        scfsolver.run();
        auto core_hole_fock_build = scfsolver.get_fock_build();
        printf("1s ionization energy % .3f eV\n",(core_hole_fock_build.first-fock_build.first)*27.2114);
      }

      return scfsolver;
    }
  }
}

std::vector<std::string> split(const std::string & input) {
  std::vector<std::string> return_value;

  std::string string;
  std::stringstream ss(input);
  while (getline(ss, string, ' '))
    if(string.size())
      return_value.push_back(string);

  return return_value;
}

std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> parse_adf_basis(const std::string & basisfile) {
  std::ifstream file(basisfile);
  if(not file.is_open())
    throw std::runtime_error("Error opening " + basisfile + "\n");

  std::string amchar("SPDFGHI");
  std::vector< std::vector<std::pair<int, double>> > functions;

  while(file.good()) {
    // Get line from file
    std::string line;
    std::getline(file, line);
    if(line.find("BASIS") != std::string::npos) {
      std::getline(file, line);
      std::vector<std::string> entries=split(line);
      while(line.find("END") == std::string::npos) {
        // Quantum number is
        std::string quantum_number(std::string(1,entries[0][0]));
        int n = std::stoi(quantum_number);
        // Angular momentum is
        int am = amchar.find(entries[0][1]);
        // Exponent is
        double expn = std::stod(entries[1]);

        if(am >= (int)functions.size()) {
          functions.resize(am+1);
        }
        functions[am].push_back(std::make_pair(n, expn));

        // Next line
        std::getline(file, line);
        entries=split(line);
      }
      break;
    }
  }

  // Sort functions
  for(size_t am=0; am<functions.size(); am++) {
    std::stable_sort(functions[am].begin(), functions[am].end(), [](const std::pair<int, double> & a, const std::pair<int, double> & b) {if(a.first < b.first) {return true;}; if(a.first > b.first) {return false;}; if(a.second > b.second) {return true;}; if(a.second < b.second) {return false;}; return false;});
  }

  std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> radial_basis;
  for(size_t am=0; am<functions.size(); am++) {
    Eigen::VectorXd exponents(functions[am].size());
    Eigen::VectorXi n_values(functions[am].size());
    for(size_t i=0; i<functions[am].size(); i++) {
      n_values(i) = functions[am][i].first;
      exponents(i) = functions[am][i].second;

      printf("%i%c %12.9f\n",n_values(i),amchar[am],exponents(i));
    }
    radial_basis.push_back(std::make_shared<const OpenOrbitalOptimizer::AtomicSolver::STOBasis>(exponents, n_values, am));
  }

  return radial_basis;
}

std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> parse_bse_json(const std::string & basisfile, int Z) {
  // Parse the JSON file
  std::ifstream basisf(basisfile);
  if(not basisf.good())
    throw std::logic_error("Error opening " + basisfile + "!\n");
  auto data = nlohmann::json::parse(basisf);

  // Need string representation of Z
  std::string Z_string;
  {
    std::ostringstream Z_ostring;
    Z_ostring << Z;
    Z_string = Z_ostring.str();
  }

  // Collect the exponents per shell
  std::vector<std::vector<std::pair<int, double>>> functions;
  for (auto & [shell_key, shell_value] : data["elements"][Z_string]["electron_shells"].items()) {
    bool cartesian = false;
    for(auto & [type_key, type_value]: shell_value["function_type"].items()) {
      if(std::string(type_value).compare("gto_cartesian")==0) {
        cartesian=true;
      }
    }

    for(auto & [am_key, am_value]: shell_value["angular_momentum"].items()) {
      int amval = am_value;
      int min_am = cartesian ? 0 : amval;
      for(int am = amval; am>=min_am; am-=2) {
        if((int)functions.size()<=am)
          functions.resize(am+1);
        for(auto & [expn_key, expn_value]: shell_value["exponents"].items()) {
          std::string expn(expn_value);
          functions[am].push_back(std::make_pair(amval+1, std::stod(expn)));
        }
      }
    }
  }
  // Sort them
  for(size_t am=0; am<functions.size(); am++) {
    std::stable_sort(functions[am].begin(), functions[am].end(), [](const std::pair<int, double> & a, const std::pair<int, double> & b) {if(a.first < b.first) {return true;}; if(a.first > b.first) {return false;}; if(a.second > b.second) {return true;}; if(a.second < b.second) {return false;}; return false;});
  }

  std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> radial_basis;
  for(size_t am=0; am<functions.size(); am++) {
    Eigen::VectorXd exponents(functions[am].size());
    Eigen::VectorXi n_values(functions[am].size());
    for(size_t i=0; i<functions[am].size(); i++) {
      n_values(i) = functions[am][i].first;
      exponents(i) = functions[am][i].second;
    }
    radial_basis.push_back(std::make_shared<const OpenOrbitalOptimizer::AtomicSolver::GTOBasis>(exponents, n_values, am));

    std::cout << "am = " << am << " exponents " << exponents.transpose() << std::endl;
    std::cout << "n values " << n_values.transpose() << std::endl;
  }

  return radial_basis;
}

int main(int argc, char **argv) {
  cmdline::parser parser;

  // full option name, no short option, description, argument required
  parser.add<int>("Z", 0, "nuclear charge", true);
  parser.add<int>("Q", 0, "atom's charge", false, 0);
  parser.add<int>("M", 0, "atom's spin multiplicity", true);
  parser.add<int>("restricted", 0, "spin restricted operation? -1 for auto", false, -1);
  parser.add<int>("Ngrid", 0, "number of radial grid points", false, 2500);
  parser.add<int>("verbosity", 0, "level of verbosity", false, 5);
  parser.add<std::string>("xfunc", 0, "exchange functional", true);
  parser.add<std::string>("cfunc", 0, "correlation functional", true);
  parser.add<std::string>("epcfunc", 0, "electron-proton correlation functional", false, "");
  parser.add<bool>("sto", 0, "Use STO instead of GTO?", false, false);
  parser.add<bool>("excitecore", 0, "Calculate core excitation?", false, false);
  parser.add<std::string>("basis", 0, "electronic basis set", true);
  parser.add<std::string>("pbasis", 0, "protonic basis set", false, "");
  parser.add<double>("lindepthresh", 0, "Linear dependence threshold", false, 1e-6);
  parser.add<double>("convthr", 0, "Convergence threshold", false, 1e-6);
  parser.add<double>("protonmass", 0, "Mass of proton in atomic units (m_p/m_e)", false, 1836.15267389); // CODATA 2014 value
  parser.parse_check(argc, argv);

  int Z = parser.get<int>("Z");
  int Q = parser.get<int>("Q");
  int M = parser.get<int>("M");
  int restricted = parser.get<int>("restricted");
  if(restricted == -1 and M == 1)
    // Automatic spin-restriction
    restricted=1;

  double linear_dependency_threshold = parser.get<double>("lindepthresh");
  double convergence_threshold = parser.get<double>("convthr");
  double proton_mass = parser.get<double>("protonmass");

  int Ngrid = parser.get<int>("Ngrid");
  int verbosity = parser.get<int>("verbosity");
  bool slater = parser.get<bool>("sto");
  bool core_excitation = parser.get<bool>("excitecore");
  std::string basisfile = parser.get<std::string>("basis");
  std::string pbasisfile = parser.get<std::string>("pbasis");

  std::string xfunc = parser.get<std::string>("xfunc");
  std::string cfunc = parser.get<std::string>("cfunc");
  std::string epcfunc = parser.get<std::string>("epcfunc");

  int x_func_id, c_func_id, epc_func_id;
  if(not xfunc.empty() and std::all_of(xfunc.begin(), xfunc.end(), ::isdigit)) {
    x_func_id = stoi(xfunc);
  } else {
    x_func_id = xc_functional_get_number(xfunc.c_str());
  }
  if(not cfunc.empty() and std::all_of(cfunc.begin(), cfunc.end(), ::isdigit)) {
    c_func_id = stoi(cfunc);
  } else {
    c_func_id = xc_functional_get_number(cfunc.c_str());
  }
  if(not epcfunc.empty() and std::all_of(epcfunc.begin(), epcfunc.end(), ::isdigit)) {
    epc_func_id = stoi(epcfunc);
  } else {
    epc_func_id = xc_functional_get_number(epcfunc.c_str());
  }
  printf("Exchange functional id is %i\n",x_func_id);
  printf("Correlation functional id is %i\n",c_func_id);
  printf("Proton correlation functional id is %i\n",epc_func_id);

  // Form the basis
  std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> radial_basis, protonic_basis;
  if(slater) {
    printf("Electronic basis\n");
    radial_basis = parse_adf_basis(basisfile);
    if(pbasisfile.size()) {
      printf("Protonic basis\n");
      protonic_basis = parse_adf_basis(pbasisfile);
    }
  } else {
    printf("Electronic basis\n");
    radial_basis = parse_bse_json(basisfile, Z);
    if(pbasisfile.size()) {
      printf("Protonic basis\n");
      protonic_basis = parse_bse_json(pbasisfile, Z);
    }
  }

  if(pbasisfile.size()) {
    OpenOrbitalOptimizer::AtomicSolver::unrestricted_neo_scf(Z, Q, M, x_func_id, c_func_id, epc_func_id, Ngrid, linear_dependency_threshold, convergence_threshold, radial_basis, protonic_basis, proton_mass, verbosity, core_excitation);
  } else {
    if(M==1) {
      OpenOrbitalOptimizer::AtomicSolver::restricted_scf(Z, Q, x_func_id, c_func_id, Ngrid, linear_dependency_threshold, convergence_threshold, radial_basis, verbosity, core_excitation);
    } else {
      OpenOrbitalOptimizer::AtomicSolver::unrestricted_scf(Z, Q, M, x_func_id, c_func_id, Ngrid, linear_dependency_threshold, convergence_threshold, radial_basis, verbosity, core_excitation);
    }
  }
  return 0;
}
