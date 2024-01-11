#include "atomicsolver.hpp"
#include <integratorxx/quadratures/all.hpp>
#include <cassert>
#include <xc.h>
#include <cmath>
#include "scfsolver.hpp"
#include <nlohmann/json.hpp>
#include "cmdline.h"

namespace OpenOrbitalOptimizer {
  // Instantiate all types of SCFSolver just to check it compiles
  template class SCFSolver<float, float>;
  template class SCFSolver<std::complex<float>, float>;
  template class SCFSolver<double, double>;
  template class SCFSolver<std::complex<double>, double>;

  namespace AtomicSolver {

    std::pair<bool, bool> eval_xc(const arma::mat & rho, const arma::mat & sigma, const arma::mat & tau, arma::vec & exc, arma::mat & vxc, arma::mat & vsigma, arma::mat & vtau, int func_id, int nspin) {
      // Initialize functional
      xc_func_type func;
      if(xc_func_init(&func, func_id, nspin) != 0) {
        std::ostringstream oss;
        oss << "Functional "<<func_id<<" not found!";
        throw std::runtime_error(oss.str());
      }

      // Energy density and potentials
      size_t N = rho.n_rows;
      exc.zeros(N);
      if(nspin==1) {
        vxc.zeros(1, N);
        vsigma.zeros(1, N);
        vtau.zeros(1, N);
      } else {
        vxc.zeros(2, N);
        vsigma.zeros(3, N);
        vtau.zeros(2, N);
      }

      // Transpose rho, sigma, and tau
      arma::mat rhot(rho.t());
      arma::mat sigmat(sigma.t());
      arma::mat taut(tau.t());

      bool gga=false, mgga=false;
      double *lapl = nullptr, *vlapl = nullptr;
      switch(func.info->family)
        {
        case XC_FAMILY_LDA:
          xc_lda_exc_vxc(&func, N, rhot.memptr(), exc.memptr(), vxc.memptr());
          break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
          xc_gga_exc_vxc(&func, N, rhot.memptr(), sigmat.memptr(), exc.memptr(), vxc.memptr(), vsigma.memptr());
          gga=true;
          break;
        case XC_FAMILY_MGGA:
        case XC_FAMILY_HYB_MGGA:
          xc_mgga_exc_vxc(&func, N, rhot.memptr(), sigmat.memptr(), lapl, taut.memptr(), exc.memptr(), vxc.memptr(), vsigma.memptr(), vlapl, vtau.memptr());
          gga=true;
          mgga=true;
          break;

        default:
          throw std::logic_error("Case not handled!\n");
        }

      // Back-transpose
      vxc = vxc.t();
      vsigma = vsigma.t();
      vtau = vtau.t();

      return std::make_pair(gga,mgga);
    }

    std::tuple<double,std::vector<arma::mat>> build_xc_unpolarized(const std::vector<std::shared_ptr<const RadialBasis>> & basis, const std::vector<arma::mat> & P, size_t N, int func_id) {
      assert(basis.size() == P.size());

      // Get radial grid
      IntegratorXX::TreutlerAhlrichs<double,double> quad(N);
      arma::vec w(arma::conv_to<arma::vec>::from(quad.weights()));
      arma::vec r(arma::conv_to<arma::vec>::from(quad.points()));
      // Angular factor
      double angfac = 4.0*M_PI;

      // Evaluate basis functions
      std::vector<arma::mat> bf(basis.size()), df(basis.size());
      for(size_t l=0;l<basis.size();l++) {
        bf[l]=basis[l]->eval_f(r);
        df[l]=basis[l]->eval_df(r);
      }

      // Electron density
      arma::mat rho(r.n_elem,1,arma::fill::zeros);
      for(size_t l=0;l<basis.size();l++)
        rho.col(0) += arma::diagvec(bf[l]*P[l]/angfac*bf[l].t());

      // Density gradient
      arma::vec drho(r.n_elem,1,arma::fill::zeros);
      for(size_t l=0;l<basis.size();l++)
        drho.col(0) += 2.0*arma::diagvec(df[l]*P[l]/angfac*bf[l].t());
      // Reduced gradient
      arma::mat sigma(arma::pow(drho,2));

      // Kinetic energy density
      arma::mat tau(r.n_elem,1,arma::fill::zeros);
      for(size_t l=0;l<basis.size();l++)
        tau.col(0) += 0.5*arma::diagvec(df[l]*P[l]/angfac*df[l].t());
      for(size_t l=1;l<basis.size();l++)
        tau.col(0) += 0.5*l*(l+1)*arma::diagvec(bf[l]*P[l]/angfac*bf[l].t())/arma::square(r);

      // Energy density and potentials
      arma::vec exc;
      arma::mat vxc;
      arma::mat vsigma;
      arma::mat vtau;
      auto ggamgga = eval_xc(rho, sigma, tau, exc, vxc, vsigma, vtau, func_id, XC_UNPOLARIZED);
      bool gga = std::get<0>(ggamgga);
      bool mgga = std::get<1>(ggamgga);

      //printf("quadrature of density yields %.10f\n",angfac*arma::dot(w, arma::square(r)%rho));
      //printf("quadrature of tau yields %.10f\n",angfac*arma::dot(w, arma::square(r)%tau));

      // xc energy
      double E = angfac*arma::dot(exc%rho, w%arma::square(r));

      // Fock matrix, LDA term
      std::vector<arma::mat> F(basis.size());
      for(size_t l=0;l<basis.size();l++)
        F[l] = bf[l].t()*arma::diagmat(w%arma::square(r)%vxc)*bf[l];
      if(gga) {
        for(size_t l=0;l<basis.size();l++) {
          arma::mat Fgga(2*df[l].t()*arma::diagmat(w%arma::square(r)%vsigma%drho)*bf[l]);
          F[l] += Fgga + Fgga.t();
        }
      }
      if(mgga) {
        for(size_t l=0;l<basis.size();l++) {
          F[l] += 0.5*df[l].t()*arma::diagmat(w%arma::square(r)%vtau)*df[l];
          if(l>0)
            F[l] += 0.5*l*(l+1)*bf[l].t()*arma::diagmat(w%vtau)*bf[l];
        }
      }

      return std::make_tuple(E,F);
    }

    std::tuple<double,std::vector<arma::mat>,std::vector<arma::mat>> build_xc_polarized(const std::vector<std::shared_ptr<const RadialBasis>> & basis, const std::vector<arma::mat> & Pa, const std::vector<arma::mat> & Pb, size_t N, int func_id) {
      assert(basis.size() == Pa.size());
      assert(basis.size() == Pb.size());

      // Get radial grid
      IntegratorXX::TreutlerAhlrichs<double,double> quad(N);
      arma::vec w(arma::conv_to<arma::vec>::from(quad.weights()));
      arma::vec r(arma::conv_to<arma::vec>::from(quad.points()));
      // Angular factor
      double angfac = 4.0*M_PI;

      // Evaluate basis functions
      std::vector<arma::mat> bf(basis.size()), df(basis.size());
      for(size_t l=0;l<basis.size();l++) {
        bf[l]=basis[l]->eval_f(r);
        df[l]=basis[l]->eval_df(r);
      }

      // Electron density; construct the transpose
      arma::mat rho(r.n_elem,2,arma::fill::zeros);
      for(size_t l=0;l<basis.size();l++) {
        rho.col(0) += arma::diagvec(bf[l]*Pa[l]/angfac*bf[l].t());
        rho.col(1) += arma::diagvec(bf[l]*Pb[l]/angfac*bf[l].t());
      }
      arma::vec rhotot(rho.col(0)+rho.col(1));

      // Density gradient; construct the transpose
      arma::mat drho(r.n_elem,2,arma::fill::zeros);
      for(size_t l=0;l<basis.size();l++) {
        drho.col(0) += 2.0*arma::diagvec(df[l]*Pa[l]/angfac*bf[l].t());
        drho.col(1) += 2.0*arma::diagvec(df[l]*Pb[l]/angfac*bf[l].t());
      }
      // Reduced gradient; construct the transpose
      arma::mat sigma(r.n_elem,3,arma::fill::zeros);
      sigma.col(0) = drho.col(0)%drho.col(0);
      sigma.col(1) = drho.col(0)%drho.col(1);
      sigma.col(2) = drho.col(1)%drho.col(1);

      // Kinetic energy density; construct the transpose
      arma::mat tau(r.n_elem,2,arma::fill::zeros);
      for(size_t l=0;l<basis.size();l++) {
        tau.col(0) += 0.5*arma::diagvec(df[l]*Pa[l]/angfac*df[l].t());
        tau.col(1) += 0.5*arma::diagvec(df[l]*Pb[l]/angfac*df[l].t());
      }
      for(size_t l=1;l<basis.size();l++) {
        tau.col(0) += 0.5*l*(l+1)*arma::diagvec(bf[l]*Pa[l]/angfac*bf[l].t())/arma::square(r);
        tau.col(1) += 0.5*l*(l+1)*arma::diagvec(bf[l]*Pb[l]/angfac*bf[l].t())/arma::square(r);
      }
      arma::vec tautot(tau.col(0)+tau.col(1));

      // Energy density and potentials
      arma::vec exc;
      arma::mat vxc;
      arma::mat vsigma;
      arma::mat vtau;
      auto ggamgga = eval_xc(rho, sigma, tau, exc, vxc, vsigma, vtau, func_id, XC_POLARIZED);
      bool gga = std::get<0>(ggamgga);
      bool mgga = std::get<1>(ggamgga);

#if 0
      printf("quadrature of alpha density yields %.10f\n",angfac*arma::dot(w, arma::square(r)%rho.col(0)));
      printf("quadrature of beta  density yields %.10f\n",angfac*arma::dot(w, arma::square(r)%rho.col(1)));
      printf("quadrature of density yields %.10f\n",angfac*arma::dot(w, arma::square(r)%rhotot));
      printf("quadrature of alpha tau yields %.10f\n",angfac*arma::dot(w, arma::square(r)%tau.col(0)));
      printf("quadrature of beta  tau yields %.10f\n",angfac*arma::dot(w, arma::square(r)%tau.col(1)));
      printf("quadrature of tau yields %.10f\n",angfac*arma::dot(w, arma::square(r)%tautot));
#endif

      // xc energy
      double E = angfac*arma::dot(exc%rhotot, w%arma::square(r));

      // Fock matrix, LDA term
      std::vector<arma::mat> Fa(basis.size()), Fb(basis.size());
      for(size_t l=0;l<basis.size();l++) {
        Fa[l] = bf[l].t()*arma::diagmat(w%arma::square(r)%vxc.col(0))*bf[l];
        Fb[l] = bf[l].t()*arma::diagmat(w%arma::square(r)%vxc.col(1))*bf[l];
      }
      if(gga) {
        for(size_t l=0;l<basis.size();l++) {
          arma::mat Fagga(df[l].t()*arma::diagmat(w%arma::square(r)%(2*vsigma.col(0)%drho.col(0) + vsigma.col(1)%drho.col(1)))*bf[l]);
          Fa[l] += Fagga + Fagga.t();
          arma::mat Fbgga(df[l].t()*arma::diagmat(w%arma::square(r)%(2*vsigma.col(2)%drho.col(1) + vsigma.col(1)%drho.col(0)))*bf[l]);
          Fb[l] += Fbgga + Fbgga.t();
        }
      }
      if(mgga) {
        for(size_t l=0;l<basis.size();l++) {
          Fa[l] += 0.5*df[l].t()*arma::diagmat(w%arma::square(r)%vtau.col(0))*df[l];
          Fb[l] += 0.5*df[l].t()*arma::diagmat(w%arma::square(r)%vtau.col(1))*df[l];
          if(l>0) {
            Fa[l] += 0.5*l*(l+1)*bf[l].t()*arma::diagmat(w%vtau.col(0))*bf[l];
            Fb[l] += 0.5*l*(l+1)*bf[l].t()*arma::diagmat(w%vtau.col(1))*bf[l];
          }
        }
      }

      return std::make_tuple(E,Fa,Fb);
    }

    std::tuple<double,std::vector<arma::mat>,std::vector<arma::mat>> build_xc_neo(const std::vector<std::shared_ptr<const RadialBasis>> & pbasis, const std::vector<arma::mat> & Pp, const std::vector<std::shared_ptr<const RadialBasis>> & ebasis, const std::vector<arma::mat> & Pe, size_t N, int func_id) {
      assert(pbasis.size() == Pp.size());
      assert(ebasis.size() == Pe.size());

      // Get radial grid
      IntegratorXX::TreutlerAhlrichs<double,double> quad(N);
      arma::vec w(arma::conv_to<arma::vec>::from(quad.weights()));
      arma::vec r(arma::conv_to<arma::vec>::from(quad.points()));
      // Angular factor
      double angfac = 4.0*M_PI;

      // Evaluate protonic and electronic basis functions
      std::vector<arma::mat> pbf(pbasis.size()), pdf(pbasis.size());
      for(size_t l=0;l<pbasis.size();l++) {
        pbf[l]=pbasis[l]->eval_f(r);
        pdf[l]=pbasis[l]->eval_df(r);
      }
      std::vector<arma::mat> ebf(ebasis.size()), edf(ebasis.size());
      for(size_t l=0;l<ebasis.size();l++) {
        ebf[l]=ebasis[l]->eval_f(r);
        edf[l]=ebasis[l]->eval_df(r);
      }

      // Proton and electron densities; construct the transposes
      arma::mat rho(r.n_elem,2,arma::fill::zeros);
      for(size_t l=0;l<pbasis.size();l++) {
        rho.col(0) += arma::diagvec(pbf[l]*Pp[l]/angfac*pbf[l].t());
      }
      for(size_t l=0;l<ebasis.size();l++) {
        rho.col(1) += arma::diagvec(ebf[l]*Pe[l]/angfac*ebf[l].t());
      }
      arma::vec rhotot(rho.col(0)+rho.col(1));

      // Density gradient; construct the transpose
      arma::mat drho(r.n_elem,2,arma::fill::zeros);
      for(size_t l=0;l<pbasis.size();l++) {
        drho.col(0) += 2.0*arma::diagvec(pdf[l]*Pp[l]/angfac*pbf[l].t());
      }
      for(size_t l=0;l<ebasis.size();l++) {
        drho.col(1) += 2.0*arma::diagvec(edf[l]*Pe[l]/angfac*ebf[l].t());
      }
      // Reduced gradient; construct the transpose
      arma::mat sigma(r.n_elem,3,arma::fill::zeros);
      sigma.col(0) = drho.col(0)%drho.col(0);
      sigma.col(1) = drho.col(0)%drho.col(1);
      sigma.col(2) = drho.col(1)%drho.col(1);

      // Kinetic energy density; construct the transpose
      arma::mat tau(r.n_elem,2,arma::fill::zeros);
      for(size_t l=0;l<pbasis.size();l++) {
        tau.col(0) += 0.5*arma::diagvec(pdf[l]*Pp[l]/angfac*pdf[l].t());
        if(l>0)
          tau.col(0) += 0.5*l*(l+1)*arma::diagvec(pbf[l]*Pp[l]/angfac*pbf[l].t())/arma::square(r);
      }
      for(size_t l=0;l<ebasis.size();l++) {
        tau.col(1) += 0.5*arma::diagvec(edf[l]*Pe[l]/angfac*edf[l].t());
        if(l>0)
          tau.col(1) += 0.5*l*(l+1)*arma::diagvec(ebf[l]*Pe[l]/angfac*ebf[l].t())/arma::square(r);
      }
      arma::vec tautot(tau.col(0)+tau.col(1));

      // Energy density and potentials
      arma::vec exc;
      arma::mat vxc;
      arma::mat vsigma;
      arma::mat vtau;
      auto ggamgga = eval_xc(rho, sigma, tau, exc, vxc, vsigma, vtau, func_id, XC_POLARIZED);
      bool gga = std::get<0>(ggamgga);
      bool mgga = std::get<1>(ggamgga);

#if 0
      printf("quadrature of proton   density yields %.10f\n",angfac*arma::dot(w, arma::square(r)%rho.col(0)));
      printf("quadrature of electron density yields %.10f\n",angfac*arma::dot(w, arma::square(r)%rho.col(1)));
      printf("quadrature of proton   tau yields %.10f\n",angfac*arma::dot(w, arma::square(r)%tau.col(0)));
      printf("quadrature of electron tau yields %.10f\n",angfac*arma::dot(w, arma::square(r)%tau.col(1)));
#endif

      // xc energy
      double E = angfac*arma::dot(exc%rhotot, w%arma::square(r));

      // Fock matrices, LDA term
      std::vector<arma::mat> Fp(pbasis.size()), Fe(ebasis.size());
      for(size_t l=0;l<pbasis.size();l++) {
        Fp[l] = pbf[l].t()*arma::diagmat(w%arma::square(r)%vxc.col(0))*pbf[l];
      }
      for(size_t l=0;l<ebasis.size();l++) {
        Fe[l] = ebf[l].t()*arma::diagmat(w%arma::square(r)%vxc.col(1))*ebf[l];
      }
      if(gga) {
        for(size_t l=0;l<pbasis.size();l++) {
          arma::mat Fpgga(pdf[l].t()*arma::diagmat(w%arma::square(r)%(2*vsigma.col(0)%drho.col(0) + vsigma.col(1)%drho.col(1)))*pbf[l]);
          Fp[l] += Fpgga + Fpgga.t();
        }
        for(size_t l=0;l<ebasis.size();l++) {
          arma::mat Fegga(edf[l].t()*arma::diagmat(w%arma::square(r)%(2*vsigma.col(2)%drho.col(1) + vsigma.col(1)%drho.col(0)))*ebf[l]);
          Fe[l] += Fegga + Fegga.t();
        }
      }
      if(mgga) {
        for(size_t l=0;l<pbasis.size();l++) {
          Fp[l] += 0.5*pdf[l].t()*arma::diagmat(w%arma::square(r)%vtau.col(0))*pdf[l];
          if(l>0)
            Fp[l] += 0.5*l*(l+1)*pbf[l].t()*arma::diagmat(w%vtau.col(0))*pbf[l];
        }
        for(size_t l=0;l<ebasis.size();l++) {
          Fe[l] += 0.5*edf[l].t()*arma::diagmat(w%arma::square(r)%vtau.col(1))*edf[l];
          if(l>0) {
            Fe[l] += 0.5*l*(l+1)*ebf[l].t()*arma::diagmat(w%vtau.col(1))*ebf[l];
          }
        }
      }

      return std::make_tuple(E,Fp,Fe);
    }

    std::tuple<double,std::vector<arma::mat>> build_J(const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & basis, const std::vector<arma::mat> & P) {
      std::vector<arma::mat> J(basis.size());
      for(size_t i=0;i<basis.size();i++) {
        J[i].zeros(basis[i]->nbf(),basis[i]->nbf());
      }

      for(size_t lin=0;lin<basis.size();lin++) {
        // Skip zero blocks
        if(arma::norm(P[lin],2) == 0.0)
          continue;
        for(size_t lout=0;lout<basis.size();lout++) {
          J[lout] += basis[lout]->coulomb(basis[lin],P[lin]);
        }
      }

      double E = 0.0;
      for(size_t l=0;l<basis.size();l++) {
        E += 0.5*arma::trace(J[l]*P[l]);
      }

      return std::make_tuple(E,J);
    }
  }
}

void restricted_scf(int Z, int Q, int x_func_id, int c_func_id, int Ngrid, double linear_dependency_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis, bool core_excitation) {
  // Form the orthogonal orbital basis
  std::vector<arma::mat> X(radial_basis.size());
  for(size_t i=0;i<X.size();i++) {
    // Overlap matrix
    arma::mat S(radial_basis[i]->overlap());

    // Normalization
    arma::vec normlz(arma::pow(arma::diagvec(S),-0.5));
    arma::mat Snorm(arma::diagmat(normlz)*S*arma::diagmat(normlz));

    // Compute X using canonical orthogonalization
    arma::vec sval;
    arma::mat svec;
    arma::eig_sym(sval, svec, Snorm);
    arma::uvec sidx(arma::find(sval>=linear_dependency_threshold));
    X[i] = svec.cols(sidx) * arma::diagmat(arma::pow(sval(sidx), -0.5));
    //sval.print("S eigenvalues");
    // Apply normalization
    X[i] = arma::diagmat(normlz) * X[i];
  }

  // Form the core Hamiltonian
  std::vector<arma::mat> T(radial_basis.size()), V(radial_basis.size());
  for(size_t i=0;i<X.size();i++) {
    T[i] = radial_basis[i]->kinetic(i);
    V[i] = - Z*radial_basis[i]->nuclear_attraction();
  }

  // Number of blocks per particle type
  arma::uvec number_of_blocks_per_particle_type({radial_basis.size()});

  arma::vec maximum_occupation(radial_basis.size());
  for(size_t l=0;l<radial_basis.size();l++)
    maximum_occupation[l] = 2*(2*l+1);

  arma::vec number_of_particles({(double) (Z-Q)});

  std::vector<std::string> block_descriptions({radial_basis.size()});
  for(size_t l=0;l<radial_basis.size();l++) {
    std::ostringstream oss;
    oss << "l=" << l;
    block_descriptions[l] = oss.str();
  }

  // Form the Fock matrix guess
  OpenOrbitalOptimizer::FockMatrix<double> fock_guess(radial_basis.size());
  for(size_t i=0;i<X.size();i++)
    fock_guess[i] = X[i].t() * (T[i]+V[i]) * X[i];

  // Fock builder
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [radial_basis, X, Ngrid, x_func_id, c_func_id, T, V](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Form the density matrix in the original basis
    std::vector<arma::mat> P(orbitals.size());
    for(size_t l=0;l<P.size();l++) {
      // In the orthonormal basis it is
      P[l] = orbitals[l] * arma::diagmat(occupations[l]) * orbitals[l].t();
      // and in the non-orthonormal basis it is
      P[l] = X[l] * P[l] * X[l].t();
    }

    // Form the non-orthonormal basis Fock matrix. Build coulomb
    auto coulomb = OpenOrbitalOptimizer::AtomicSolver::build_J(radial_basis, P);
    // Build exchange
    auto exchange = OpenOrbitalOptimizer::AtomicSolver::build_xc_unpolarized(radial_basis, P, Ngrid, x_func_id);
    // and correlation
    auto correlation = OpenOrbitalOptimizer::AtomicSolver::build_xc_unpolarized(radial_basis, P, Ngrid, c_func_id);

    // Collect the Fock matrix in the orthonormal basis
    std::vector<arma::mat> fock(orbitals.size());
    for(size_t l=0; l<fock.size(); l++) {
      fock[l] = X[l].t() * (T[l] + V[l] + std::get<1>(coulomb)[l] + std::get<1>(exchange)[l] + std::get<1>(correlation)[l]) * X[l];

      arma::Mat J(std::get<1>(coulomb)[l]);
      arma::Mat K(std::get<1>(exchange)[l]);
      double nucasymm(arma::norm(V[l]-V[l].t(),2));
      double kinasymm(arma::norm(T[l]-T[l].t(),2));
      double coulasymm(arma::norm(J-J.t(),2));
      double exchasymm(arma::norm(K-K.t(),2));
      //printf("l=%i V asymm %e T asymm %e J asymm %e K asymm %e\n",l,nucasymm,kinasymm,coulasymm,exchasymm);
    }

    // Calculate energy terms
    double Ekin = 0.0;
    for(size_t l=0;l<P.size();l++)
      Ekin += arma::trace(T[l]*P[l]);

    double Enuc = 0.0;
    for(size_t l=0;l<P.size();l++)
      Enuc += arma::trace(V[l]*P[l]);

    double Ej = std::get<0>(coulomb);
    double Ex = std::get<0>(exchange);
    double Ec = std::get<0>(correlation);
    double Etot = Ekin+Enuc+Ej+Ex+Ec;

#if 0
    printf("Kinetic energy  % .10f\n",Ekin);
    printf("Nuclear energy  % .10f\n",Enuc);
    printf("Coulomb energy  % .10f\n",Ej);
    printf("Exchange energy % .10f\n",Ex);
    printf("Correlation energy % .10f\n",Ec);
    printf("Total energy    % .10f\n",Etot);
#endif

    return std::make_pair(Etot, fock);
  };

  // Initialize SCF solver
  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(fock_guess);
  scfsolver.run();
  // scfsolver.brute_force_search_for_lowest_configuration();

  if(core_excitation) {
    // Form core-excited state
    auto density_matrix = scfsolver.get_solution();
    auto orbitals =  density_matrix.first;
    auto occupations =  density_matrix.second;
    auto fock_build = scfsolver.get_fock_build();

    // Decrease occupation of 1s orbital
    occupations[0](0) = 0.0;
    scfsolver.set_frozen_occupations(true);
    scfsolver.initialize_with_orbitals(orbitals, occupations);
    scfsolver.run();
    auto core_hole_fock_build = scfsolver.get_fock_build();
    printf("1s double ionization energy % .3f eV\n",(core_hole_fock_build.first-fock_build.first)*27.2114);
  }
}

void unrestricted_scf(int Z, int Q, int M, int x_func_id, int c_func_id, int Ngrid, double linear_dependency_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis, bool core_excitation) {
  // Form the orthogonal orbital basis
  std::vector<arma::mat> X(2*radial_basis.size());
  for(size_t i=0;i<radial_basis.size();i++) {
    // Overlap matrix
    arma::mat S(radial_basis[i]->overlap());

    // Normalization
    arma::vec normlz(arma::pow(arma::diagvec(S),-0.5));
    arma::mat Snorm(arma::diagmat(normlz)*S*arma::diagmat(normlz));

    // Compute X using canonical orthogonalization
    arma::vec sval;
    arma::mat svec;
    arma::eig_sym(sval, svec, Snorm);
    arma::uvec sidx(arma::find(sval>=linear_dependency_threshold));
    X[i] = svec.cols(sidx) * arma::diagmat(arma::pow(sval(sidx), -0.5));
    //sval.print("S eigenvalues");
    // Apply normalization
    X[i] = arma::diagmat(normlz) * X[i];
  }

  // Form the core Hamiltonian
  std::vector<arma::mat> T(2*radial_basis.size()), V(2*radial_basis.size());
  for(size_t i=0;i<radial_basis.size();i++) {
    T[i] = radial_basis[i]->kinetic(i);
    V[i] = - Z*radial_basis[i]->nuclear_attraction();
  }

  // Repeat blocks for second spin channel
  for(size_t i=0;i<radial_basis.size();i++) {
    X[i+radial_basis.size()] = X[i];
    T[i+radial_basis.size()] = T[i];
    V[i+radial_basis.size()] = V[i];
  }

  // Number of blocks per particle type
  arma::uvec number_of_blocks_per_particle_type({radial_basis.size(),radial_basis.size()});

  arma::vec maximum_occupation(2*radial_basis.size());
  for(size_t l=0;l<radial_basis.size();l++)
    maximum_occupation[l] = 2*l+1;
  for(size_t i=0;i<radial_basis.size();i++) {
    maximum_occupation[i+radial_basis.size()] = maximum_occupation[i];
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
  arma::vec number_of_particles({(double) Nela,(double) Nelb});

  std::vector<std::string> block_descriptions({2*radial_basis.size()});
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

  // Form the Fock matrix guess
  OpenOrbitalOptimizer::FockMatrix<double> fock_guess(X.size());
  for(size_t i=0;i<X.size();i++)
    fock_guess[i] = X[i].t() * (T[i]+V[i]) * X[i];

  // Fock builder
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [radial_basis, X, Ngrid, x_func_id, c_func_id, T, V](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    // Form the spin-up and spin-down density matrices in the original basis
    assert(orbitals.size()%2==0);
    size_t Nblocks = orbitals.size()/2;
    std::vector<arma::mat> Pa(Nblocks), Pb(Nblocks), Ptot(Nblocks);
    for(size_t iblock=0;iblock<Nblocks;iblock++) {
      size_t ablock = iblock;
      // In the orthonormal basis it is
      Pa[iblock] = orbitals[ablock] * arma::diagmat(occupations[ablock]) * orbitals[ablock].t();
      // and in the non-orthonormal basis it is
      Pa[iblock] = X[iblock] * Pa[iblock] * X[iblock].t();
      size_t bblock = iblock+Nblocks;
      // In the orthonormal basis it is
      Pb[iblock] = orbitals[bblock] * arma::diagmat(occupations[bblock]) * orbitals[bblock].t();
      // and in the non-orthonormal basis it is
      Pb[iblock] = X[iblock] * Pb[iblock] * X[iblock].t();

      // Since we use same X for both spin channels, total density is
      Ptot[iblock] = Pa[iblock] + Pb[iblock];
    }

    // Form the non-orthonormal basis Fock matrix. Build coulomb
    auto coulomb = OpenOrbitalOptimizer::AtomicSolver::build_J(radial_basis, Ptot);
    // Build exchange
    auto exchange = OpenOrbitalOptimizer::AtomicSolver::build_xc_polarized(radial_basis, Pa, Pb, Ngrid, x_func_id);
    // and correlation
    auto correlation = OpenOrbitalOptimizer::AtomicSolver::build_xc_polarized(radial_basis, Pa, Pb, Ngrid, c_func_id);

    // Collect the Fock matrix in the orthonormal basis
    std::vector<arma::mat> fock(orbitals.size());
    for(size_t iblock=0; iblock<Nblocks; iblock++) {
      // Spin-up Fock
      size_t ablock = iblock, bblock = iblock+Nblocks;
      fock[ablock] = X[ablock].t() * (T[ablock] + V[ablock] + std::get<1>(coulomb)[iblock] + std::get<1>(exchange)[iblock] + std::get<1>(correlation)[iblock]) * X[ablock];
      fock[bblock] = X[bblock].t() * (T[bblock] + V[bblock] + std::get<1>(coulomb)[iblock] + std::get<2>(exchange)[iblock] + std::get<2>(correlation)[iblock]) * X[bblock];
    }

    // Calculate energy terms
    double Ekin = 0.0;
    for(size_t l=0;l<Ptot.size();l++)
      Ekin += arma::trace(T[l]*Ptot[l]);

    double Enuc = 0.0;
    for(size_t l=0;l<Ptot.size();l++)
      Enuc += arma::trace(V[l]*Ptot[l]);

    double Ej = std::get<0>(coulomb);
    double Ex = std::get<0>(exchange);
    double Ec = std::get<0>(correlation);
    double Etot = Ekin+Enuc+Ej+Ex+Ec;

#if 0
    printf("Kinetic energy     % .10f\n",Ekin);
    printf("Nuclear energy     % .10f\n",Enuc);
    printf("Coulomb energy     % .10f\n",Ej);
    printf("Exchange energy    % .10f\n",Ex);
    printf("Correlation energy % .10f\n",Ec);
    printf("Total energy       % .10f\n",Etot);
#endif

    return std::make_pair(Etot, fock);
  };

  // Initialize SCF solver
  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(fock_guess);
  scfsolver.run();
  //scfsolver.brute_force_search_for_lowest_configuration();

  if(core_excitation) {
    // Form core-excited state
    auto density_matrix = scfsolver.get_solution();
    auto orbitals =  density_matrix.first;
    auto occupations =  density_matrix.second;
    auto fock_build = scfsolver.get_fock_build();

    // Decrease occupation of 1s orbital
    occupations[0](0) = 0.0;
    scfsolver.set_frozen_occupations(true);
    scfsolver.initialize_with_orbitals(orbitals, occupations);
    scfsolver.run();
    auto core_hole_fock_build = scfsolver.get_fock_build();
    printf("1s ionization energy % .3f eV\n",(core_hole_fock_build.first-fock_build.first)*27.2114);
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

        if(am >= functions.size()) {
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
    arma::vec exponents(functions[am].size());
    arma::ivec n_values(functions[am].size());
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
  auto data = nlohmann::json::parse(basisf);

  // Need string representation of Z
  std::string Z_string;
  {
    std::ostringstream Z_ostring;
    Z_ostring << Z;
    Z_string = Z_ostring.str();
  }

  // Collect the exponents per shell
  std::vector<std::vector<double>> shell_exponents;
  for (auto & [shell_key, shell_value] : data["elements"][Z_string]["electron_shells"].items()) {
    for(auto & [am_key, am_value]: shell_value["angular_momentum"].items()) {
      int am = am_value;
      if(shell_exponents.size()<=am)
        shell_exponents.resize(am+1);
      for(auto & [expn_key, expn_value]: shell_value["exponents"].items()) {
        std::string expn(expn_value);
        shell_exponents[am].push_back(std::stod(expn));
      }
    }
  }
  // Sort them
  for(auto shell : shell_exponents)
    std::stable_sort(shell.begin(), shell.end());

  std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> radial_basis;
  for(size_t am=0; am<shell_exponents.size(); am++) {
    arma::vec exponents = arma::conv_to<arma::vec>::from(shell_exponents[am]);
    arma::ivec n_values = (am+1)*arma::ones<arma::ivec>(exponents.n_elem);
    radial_basis.push_back(std::make_shared<const OpenOrbitalOptimizer::AtomicSolver::GTOBasis>(exponents, n_values, am));

    std::cout << "am = " << am << " exponents";
    exponents.t().print();
    n_values.t().print("n values");
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
  parser.add<std::string>("xfunc", 0, "exchange functional", true);
  parser.add<std::string>("cfunc", 0, "correlation functional", true);
  parser.add<std::string>("epcfunc", 0, "electron-proton correlation functional", false, "");
  parser.add<bool>("sto", 0, "Use STO instead of GTO?", false, false);
  parser.add<bool>("excitecore", 0, "Calculate core excitation?", false, false);
  parser.add<std::string>("basis", 0, "electronic basis set", true);
  parser.add<std::string>("pbasis", 0, "protonic basis set", false, "");
  parser.add<double>("lindepthresh", 0, "Linear dependence threshold", false, 1e-6);
  parser.parse_check(argc, argv);

  int Z = parser.get<int>("Z");
  int Q = parser.get<int>("Q");
  int M = parser.get<int>("M");
  int restricted = parser.get<int>("restricted");
  if(restricted == -1 and M == 1)
    // Automatic spin-restriction
    restricted=1;

  double linear_dependency_threshold = parser.get<double>("lindepthresh");
  int Ngrid = parser.get<int>("Ngrid");
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

  // Form the basis
  std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> radial_basis;
  if(slater) {
    radial_basis = parse_adf_basis(basisfile);
  } else {
    radial_basis = parse_bse_json(basisfile, Z);
  }

  if(M==1) {
    restricted_scf(Z, Q, x_func_id, c_func_id, Ngrid, linear_dependency_threshold, radial_basis, core_excitation);
  } else {
    unrestricted_scf(Z, Q, M, x_func_id, c_func_id, Ngrid, linear_dependency_threshold, radial_basis, core_excitation);
  }
  return 0;
}
