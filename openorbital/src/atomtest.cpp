#include "atomicsolver.hpp"
#include <integratorxx/quadratures/all.hpp>
#include <cassert>
#include <xc.h>
#include <cmath>
#include "scfsolver.hpp"

namespace OpenOrbitalOptimizer {
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

void restricted_scf(int Z, int Q, int x_func_id, int Ngrid, double linear_dependency_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis) {
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
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [radial_basis, X, Ngrid, x_func_id, T, V](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
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

    // Collect the Fock matrix in the orthonormal basis
    std::vector<arma::mat> fock(orbitals.size());
    for(size_t l=0; l<fock.size(); l++) {
      fock[l] = X[l].t() * (T[l] + V[l] + std::get<1>(coulomb)[l] + std::get<1>(exchange)[l]) * X[l];

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
    double Exc = std::get<0>(exchange);
    double Etot = Ekin+Enuc+Ej+Exc;
#if 0
    printf("Kinetic energy  % .10f\n",Ekin);
    printf("Nuclear energy  % .10f\n",Enuc);
    printf("Coulomb energy  % .10f\n",Ej);
    printf("Exchange energy % .10f\n",Exc);
    printf("Total energy    % .10f\n",Etot);
#endif
    return std::make_pair(Etot, fock);
  };

  // Initialize SCF solver
  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(fock_guess);
  scfsolver.run();
}

void unrestricted_scf(int Z, int Q, int M, int x_func_id, int Ngrid, double linear_dependency_threshold, const std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> & radial_basis) {
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
  OpenOrbitalOptimizer::FockBuilder<double, double> fock_builder = [radial_basis, X, Ngrid, x_func_id, T, V](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
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

    // Collect the Fock matrix in the orthonormal basis
    std::vector<arma::mat> fock(orbitals.size());
    for(size_t iblock=0; iblock<Nblocks; iblock++) {
      // Spin-up Fock
      size_t ablock = iblock, bblock = iblock+Nblocks;
      fock[ablock] = X[ablock].t() * (T[ablock] + V[ablock] + std::get<1>(coulomb)[iblock] + std::get<1>(exchange)[iblock]) * X[ablock];
      fock[bblock] = X[bblock].t() * (T[bblock] + V[bblock] + std::get<1>(coulomb)[iblock] + std::get<2>(exchange)[iblock]) * X[bblock];
    }
    
    // Calculate energy terms
    double Ekin = 0.0;
    for(size_t l=0;l<Ptot.size();l++)
      Ekin += arma::trace(T[l]*Ptot[l]);
    
    double Enuc = 0.0;
    for(size_t l=0;l<Ptot.size();l++)
      Enuc += arma::trace(V[l]*Ptot[l]);
  
    double Ej = std::get<0>(coulomb);
    double Exc = std::get<0>(exchange);
    double Etot = Ekin+Enuc+Ej+Exc;
#if 0
    printf("Kinetic energy  % .10f\n",Ekin);
    printf("Nuclear energy  % .10f\n",Enuc);
    printf("Coulomb energy  % .10f\n",Ej);
    printf("Exchange energy % .10f\n",Exc);
    printf("Total energy    % .10f\n",Etot);
#endif
    return std::make_pair(Etot, fock);
  };
  
  // Initialize SCF solver
  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  scfsolver.initialize_with_fock(fock_guess);
  scfsolver.run();
}

int main(void) {
  // Ne TZ2P basis from ADF (no polarization)
  int Z = 10;
  //int Z = 2;

  int x_func_id = 202;

  double linear_dependency_threshold = 1e-6;
  int Ngrid = 2500;
  bool slater=true;

  arma::vec ne_s_exp({12.45, 8.7, 3.65, 2.20, 1.55}); // ne
  arma::ivec ne_s_n({1, 1, 2, 2, 2});
  //arma::vec ne_s_exp({3.029062440664153, 1.687500007289023, 0.9401114471499884, 0.5237389802932741}); // he
  //arma::ivec ne_s_n({1, 1, 1, 1});
  //arma::vec ne_s_exp(arma::logspace<arma::vec>(-3,7,21));
  //arma::ivec ne_s_n(arma::ones<arma::ivec>(ne_s_exp.size()));
  //arma::vec ne_s_exp(arma::logspace<arma::vec>(-3,3,21));
  //arma::ivec ne_s_n(arma::ones<arma::ivec>(ne_s_exp.size()));
  //if(slater)
  //  ne_s_exp *= Z;
  ne_s_exp.print("s exponents");
  
  arma::vec ne_p_exp({5.50, 2.85, 1.45});
  arma::ivec ne_p_n({2,2,2});
  //arma::vec ne_p_exp(arma::logspace<arma::vec>(-2,2,15));
  //arma::vec ne_p_exp(ne_s_exp);
  //arma::ivec ne_p_n(2*arma::ones<arma::ivec>(ne_p_exp.size()));
  //if(slater)
  //  ne_p_exp *= Z;
  ne_p_exp.print("p exponents");
  
  // Form the basis
  std::vector<std::shared_ptr<const OpenOrbitalOptimizer::AtomicSolver::RadialBasis>> radial_basis;
  if(slater) {
    radial_basis.push_back(std::make_shared<const OpenOrbitalOptimizer::AtomicSolver::STOBasis>(ne_s_exp,ne_s_n,0));
    radial_basis.push_back(std::make_shared<const OpenOrbitalOptimizer::AtomicSolver::STOBasis>(ne_p_exp,ne_p_n,1));
  } else {
    radial_basis.push_back(std::make_shared<const OpenOrbitalOptimizer::AtomicSolver::GTOBasis>(ne_s_exp,ne_s_n,0));
    radial_basis.push_back(std::make_shared<const OpenOrbitalOptimizer::AtomicSolver::GTOBasis>(ne_p_exp,ne_p_n,1));
  }

  int Q=0;
  int M=1;
  if(M==1) {
    restricted_scf(Z, Q, x_func_id, Ngrid, linear_dependency_threshold, radial_basis);
  } else {
    unrestricted_scf(Z, Q, M, x_func_id, Ngrid, linear_dependency_threshold, radial_basis);
 }


  return 0;
}
