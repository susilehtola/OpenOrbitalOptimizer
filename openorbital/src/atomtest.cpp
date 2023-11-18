#include "atomicsolver.hpp"

int main(void) {
  // Ne TZ2P basis from ADF (no polarization)
  //int Z = 10;
  int Z = 1;
  //arma::vec ne_s_exp({12.45, 8.7, 3.65, 2.20, 1.55});
  // arma::ivec ne_s_n({1, 1, 2, 2, 2});
  arma::vec ne_s_exp(arma::logspace<arma::vec>(-3,3,25));
  //arma::vec ne_s_exp({1.0});
  arma::ivec ne_s_n(arma::ones<arma::ivec>(ne_s_exp.size()));
  ne_s_exp.print("s exponents");
  
  //arma::vec ne_p_exp({5.50, 2.85, 1.45});
  // arma::ivec ne_p_n({2,2,2});
  //arma::vec ne_p_exp(arma::logspace<arma::vec>(-2,2,15));
  arma::vec ne_p_exp(ne_s_exp);
  arma::ivec ne_p_n(2*arma::ones<arma::ivec>(ne_p_exp.size()));
  ne_p_exp.print("p exponents");
  
  // Form the basis
#if 1
  std::vector<openorbital::atomicsolver::STOBasis> radial_basis;
  radial_basis.push_back(openorbital::atomicsolver::STOBasis(ne_s_exp,ne_s_n,0));
  radial_basis.push_back(openorbital::atomicsolver::STOBasis(ne_p_exp,ne_p_n,1));
#else
  std::vector<openorbital::atomicsolver::GTOBasis> radial_basis;
  radial_basis.push_back(openorbital::atomicsolver::GTOBasis(ne_s_exp,ne_s_n,0));
  radial_basis.push_back(openorbital::atomicsolver::GTOBasis(ne_p_exp,ne_p_n,1));
#endif
  
  // Set up SCF spaces
  arma::uvec number_of_blocks_per_particle_type({2});
  arma::vec maximum_occupation({2,6});
  arma::vec number_of_particles({10});

  // Form the orthogonal orbital basis
  std::vector<arma::mat> X(radial_basis.size());
  for(size_t i=0;i<X.size();i++) {
    // Overlap matrix
    arma::mat S(radial_basis[i].overlap());

    // Normalization
    arma::vec normlz(arma::pow(arma::diagvec(S),-0.5));
    arma::mat Snorm(arma::diagmat(normlz)*S*arma::diagmat(normlz));

    // Compute X using symmetric orthogonalization
    arma::vec sval;
    arma::mat svec;
    arma::eig_sym(sval, svec, Snorm);
    X[i] = svec * arma::diagmat(arma::pow(sval, -0.5)) * svec.t();
    sval.print("S eigenvalues");
    // Apply normalization
    X[i] = arma::diagmat(normlz) * X[i];
  }

  // Form the core Hamiltonian
  std::vector<arma::mat> Hcore(radial_basis.size());
  for(size_t i=0;i<X.size();i++)
    Hcore[i] = radial_basis[i].kinetic(i) - Z*radial_basis[i].nuclear_attraction();

  // Form the Fock matrix guess
  OpenOrbitalOptimizer::FockMatrix<double> fock_guess(radial_basis.size());
  for(size_t i=0;i<X.size();i++)
    fock_guess[i] = X[i].t() * Hcore[i] * X[i];

  // Diagonalize Fock matrix
  for(size_t i=0;i<X.size();i++) {
    arma::vec E;
    arma::mat C;
    arma::eig_sym(E,C,fock_guess[i]);
    printf("l=%lu orbital energies\n",i);
    E.print();

    // Test Coulomb
    if(i==0) {
      arma::mat Corig(X[i]*C);
      arma::mat Ptest(Corig.col(0)*Corig.col(0).t());      
      arma::mat J(radial_basis[0].coulomb(radial_basis[0],Ptest));
      arma::mat Jmo(C.t()*J*C);
            
      double Esic = 0.5*arma::trace(Ptest*J);
      printf("Coulomb self-energy % .6f\n",Esic);
      Ptest.print("Ptest");
      Jmo.print("Jmo");
      C.col(0).print("Lowest orbital ocefficients");
    }
  }

  return 0;
}
