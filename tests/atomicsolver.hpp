#include <openorbitaloptimizer/scfsolver.hpp>
#include <armadillo>
#include <cassert>
#include <memory>

namespace OpenOrbitalOptimizer {
  namespace AtomicSolver {

    enum RadialBasisType {
      STOBASIS,
      GTOBASIS
    };

    class RadialBasis {
    protected:
      /// Angular momentum of the function
      int angular_momentum_;

      inline double Vn(double n, double x) const {
        return std::tgamma(n+1)/std::pow(x,n+1);
      }

      inline double Wn(int n, double x) const {
        return (n-1)/x;
      }

#if 0
      inline double Enk(int n, int k, double x, int increment=1) const {
        double E=0.0; // E^n_0 = 0
        for(int ik=1;ik<=k;ik+=increment)
          E=ik*(1.0+E)/((n-ik+1.0)*x);
        return E;
      }
#else
      inline double Enk(int n, int k, double x) const {
        double E=0.0; // E^n_0 = 0
        for(int ik=1;ik<=k;ik++)
          E=ik*(1.0+E)/((n-ik+1.0)*x);
        return E;
      }

      inline double binomial(double n, double k) const {
        double binomial = (std::tgamma(n+1.0)/std::tgamma(n-k+1.0))/std::tgamma(k+1.0);
        return binomial;
      }

      inline double Enk(double n, double k, double x) const {
        double E=0.0;
        for(int j=0;j<k;j++)
          E += binomial(n,j)*std::pow(x,j);
        E /= binomial(n,k)*std::pow(x,k);
        return E;
      }
#endif
      /// Type of this radial basis
      RadialBasisType type_;
      /// Coulomb integral
      virtual double Rmnv(int m, int n, int v, double x, double y) const {throw std::logic_error("Not implemented!\n");}

    public:
      RadialBasis(int angular_momentum) : angular_momentum_(angular_momentum) {};
      /// Get radial basis type
      RadialBasisType get_type() const {return type_;}
      /// Evaluate overlap matrix
      virtual arma::mat overlap() const=0;
      /// Evaluate kinetic energy matrix
      virtual arma::mat kinetic(int l) const=0;
      /// Evaluate nuclear attraction matrix
      virtual arma::mat nuclear_attraction() const=0;
      /// Evaluate radial basis functions
      virtual arma::mat eval_f(const arma::vec & x) const {throw std::logic_error("Not implemented!\n");};
      /// Evaluate radial basis functions' derivatives
      virtual arma::mat eval_df(const arma::vec & x) const {throw std::logic_error("Not implemented!\n");};
      /// Build Coulomb matrix
      virtual arma::mat coulomb(const std::shared_ptr<const RadialBasis> & other, const arma::mat & Pother) const=0;
      /// Return number of basis functions
      virtual size_t nbf() const=0;
    };

    class STOBasis : public RadialBasis {
    private:
      /// STO exponents
      arma::vec zeta_;
      /// STO principal quantum numbers
      arma::ivec n_;

      /// Evaluate two-electron integral
      inline double Rmnv(int m, int n, int v, double x, double y) const {
        double value = std::tgamma(m+n)/(x*y*std::pow(x+y,m+n-1))*(1+Enk(m+n-1,n-v-1,y/x)+Enk(m+n-1,m-v-1,x/y));
        //printf("Rmnv(%i,%i,%i,%e,%e) = %e\n",m,n,v,x,y,value);
        return value;
      }
      /// Pairs of basis functions
      std::vector<std::pair<size_t,size_t>> basis_function_pairs() const {
        std::vector<std::pair<size_t,size_t>> list;
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++)
            list.push_back(std::make_pair(i,j));
        return list;
      }
    public:
      /// Constructor
      STOBasis(const arma::vec & zeta, const arma::ivec & n, int angular_momentum) : zeta_(zeta), n_(n), RadialBasis(angular_momentum) {
        type_ = STOBASIS;
        if(zeta_.n_elem != n_.n_elem)
          throw std::logic_error("zeta and n need to be of same size!\n");
      }
      /// Evaluate overlap matrix
      arma::mat overlap() const override {
        auto list(basis_function_pairs());
        arma::mat S(zeta_.n_elem, zeta_.n_elem);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          // Pitzer, page 244
          S(i,j) = S(j,i) = Vn(n_(i)+n_(j), zeta_(i)+zeta_(j));
          //printf("S(%i,%i) %e\n",i,j,S(i,j));
        }
        return S;
      };
      /// Evaluate kinetic energy matrix
      arma::mat kinetic(int am) const override {
        auto list(basis_function_pairs());
        arma::mat T(zeta_.n_elem, zeta_.n_elem);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          // Pitzer, page 244
          T(i,j) = T(j,i) = 0.5*zeta_(i)*zeta_(j)*(
                                                   Wn(n_(i)-am, zeta_(i)) * Wn(n_(j)-am, zeta_(j)) * Vn(n_(i)+n_(j)-2, zeta_(i)+zeta_(j))
                                                   - (Wn(n_(i)-am, zeta_(i)) + Wn(n_(j)-am, zeta_(j))) * Vn(n_(i)+n_(j)-1, zeta_(i)+zeta_(j))
                                                   + Vn(n_(i)+n_(j), zeta_(i) + zeta_(j))
                                                   );
          //printf("T(%i,%i) %e\n",i,j,T(i,j));
        }
        return T;
      }
      /// Evaluate nuclear attraction matrix
      arma::mat nuclear_attraction() const override {
        auto list(basis_function_pairs());
        arma::mat V(zeta_.n_elem, zeta_.n_elem);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          // Pitzer, page 244
          V(i,j) = V(j,i) = Vn(n_(i)+n_(j)-1, zeta_(i)+zeta_(j));
          //printf("V(%i,%i) %e\n",i,j,V(i,j));
        }
        return V;
      }
      /// Evaluate basis functions
      arma::mat eval_f(const arma::vec & x) const override {
        arma::mat f(x.n_elem, zeta_.n_elem, arma::fill::zeros);
#pragma omp parallel for collapse(2)
        for(size_t iz=0; iz<zeta_.n_elem; iz++) {
          for(size_t ix=0; ix<x.n_elem; ix++) {
            f(ix,iz) = std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix));
          }
        }
        return f;
      }
      /// Evaluate basis functions
      arma::mat eval_df(const arma::vec & x) const override {
        arma::mat df(x.n_elem, zeta_.n_elem, arma::fill::zeros);
#pragma omp parallel for collapse(2)
        for(size_t iz=0; iz<zeta_.n_elem; iz++) {
          for(size_t ix=0; ix<x.n_elem; ix++) {
            df(ix,iz) = -zeta_(iz) * std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix));
            if(n_(iz)>1) {
              df(ix,iz) += (n_(iz)-1) * std::pow(x(ix),n_(iz)-2) * std::exp(-zeta_(iz)*x(ix));
            }
          }
        }
        return df;
      }
      /// Evaluate Coulomb matrix
      arma::mat coulomb(const STOBasis & other, const arma::mat & Pother) const {
        auto list(basis_function_pairs());

        arma::mat J(zeta_.n_elem, zeta_.n_elem, arma::fill::zeros);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          double Jij = 0.0;
          for(size_t l=0;l<Pother.n_cols;l++)
            for(size_t k=0;k<Pother.n_rows;k++)
              Jij += Pother(k,l) * Rmnv(n_(i)+n_(j), other.n_(k)+other.n_(l), 0, zeta_(i)+zeta_(j), other.zeta_(k)+other.zeta_(l));
          J(i,j) = J(j,i) = Jij;
        }
        return J;
      }
      /// Wrapper for the above
      arma::mat coulomb(const std::shared_ptr<const RadialBasis> & other, const arma::mat & Pother) const override {
        assert(other->get_type() == STOBASIS);
        auto pother = std::dynamic_pointer_cast<const STOBasis>(other);
        return coulomb(*pother, Pother);
      }
      /// Return number of basis functions
      size_t nbf() const override {
        return zeta_.n_elem;
      }
    };

    class GTOBasis : public RadialBasis {
    private:
      /// GTO exponents
      arma::vec zeta_;
      /// GTO principal quantum numbers
      arma::ivec n_;

      /// Evaluate two-electron integral
      inline double Rmnv(int m, int n, int v, double x, double y) const {
        double value = std::tgamma((m+n-1)/2.0)/(x*y*std::pow(x+y,(m+n-3)/2.0))*(1+Enk(0.5*(m+n-3),0.5*(n-v-2),y/x)+Enk(0.5*(m+n-3),0.5*(m-v-2),x/y))/4.0;
        //printf("Rmnv(%i,%i,%i,%e,%e) = %e\n",m,n,v,x,y,value);
        return value;
      }
      /// Pairs of basis functions
      std::vector<std::pair<size_t,size_t>> basis_function_pairs() const {
        std::vector<std::pair<size_t,size_t>> list;
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++)
            list.push_back(std::make_pair(i,j));
        return list;
      }
    public:
      /// Constructor
      GTOBasis(const arma::vec & zeta, const arma::ivec & n, int angular_momentum) : zeta_(zeta), n_(n), RadialBasis(angular_momentum) {
        type_ = GTOBASIS;
        if(zeta_.n_elem != n_.n_elem)
          throw std::logic_error("zeta and n need to be of same size!\n");
      };
      /// Evaluate overlap matrix
      arma::mat overlap() const override {
        auto list(basis_function_pairs());
        arma::mat S(zeta_.n_elem, zeta_.n_elem);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          // Pitzer, page 244. Factor one half is missing in paper
          S(i,j) = S(j,i) = 0.5*Vn(0.5*(n_(i)+n_(j)-1),zeta_(i)+zeta_(j));
        }
        return S;
      };
      /// Evaluate kinetic energy matrix
      arma::mat kinetic(int l) const override {
        auto list(basis_function_pairs());
        arma::mat T(zeta_.n_elem, zeta_.n_elem);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          // Pitzer, page 244. Factor one half is missing in paper
          T(i,j) = T(j,i) = zeta_(i)*zeta_(j)*(
                                               Wn(n_(i)-l,2*zeta_(i)) * Wn(n_(j)-l,2*zeta_(j)) * Vn(0.5*(n_(i)+n_(j)-3),zeta_(i)+zeta_(j))
                                               - (Wn(n_(i)-l,2*zeta_(i)) + Wn(n_(j)-l,2*zeta_(j))) * Vn(0.5*(n_(i)+n_(j)-1),zeta_(i)+zeta_(j))
                                               + Vn(0.5*(n_(i)+n_(j)+1),zeta_(i)+zeta_(j))
                                               );
        }
        return T;
      }
      /// Evaluate nuclear attraction matrix
      arma::mat nuclear_attraction() const override {
        auto list(basis_function_pairs());
        arma::mat V(zeta_.n_elem, zeta_.n_elem);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          // Pitzer, page 244. Factor one half is missing in paper
          V(i,j) = V(j,i) = 0.5*Vn(0.5*(n_(i)+n_(j)-2),zeta_(i)+zeta_(j));
        }
        return V;
      }
      /// Evaluate basis functions
      arma::mat eval_f(const arma::vec & x) const override {
        arma::mat f(x.n_elem, zeta_.n_elem, arma::fill::zeros);
#pragma omp parallel for collapse(2)
        for(size_t iz=0; iz<zeta_.n_elem; iz++) {
          for(size_t ix=0; ix<x.n_elem; ix++) {
            f(ix,iz) = std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix)*x(ix));
          }
        }
        return f;
      }
      /// Evaluate basis functions
      arma::mat eval_df(const arma::vec & x) const override {
        arma::mat df(x.n_elem, zeta_.n_elem, arma::fill::zeros);
#pragma omp parallel for collapse(2)
        for(size_t iz=0; iz<zeta_.n_elem; iz++) {
          for(size_t ix=0; ix<x.n_elem; ix++) {
            df(ix,iz) = -2.0 * zeta_(iz) * x(ix) * std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix)*x(ix));
            if(n_(iz)>1) {
              df(ix,iz) += (n_(iz)-1) * std::pow(x(ix),n_(iz)-2) * std::exp(-zeta_(iz)*x(ix)*x(ix));
            }
          }
        }
        return df;
      }
      /// Evaluate Coulomb matrix
      arma::mat coulomb(const GTOBasis & other, const arma::mat & Pother) const {
        auto list(basis_function_pairs());

        arma::mat J(zeta_.n_elem, zeta_.n_elem, arma::fill::zeros);
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          double Jij = 0.0;
          for(size_t l=0;l<Pother.n_cols;l++)
            for(size_t k=0;k<Pother.n_rows;k++)
              Jij += Pother(k,l) * Rmnv(n_(i)+n_(j), other.n_(k)+other.n_(l), 0, zeta_(i)+zeta_(j), other.zeta_(k)+other.zeta_(l));
          J(i,j) = J(j,i) = Jij;
        }
        return J;
      }
      /// Wrapper for the above
      arma::mat coulomb(const std::shared_ptr<const RadialBasis> & other, const arma::mat & Pother) const override {
        assert(other->get_type() == GTOBASIS);
        auto pother = std::dynamic_pointer_cast<const GTOBasis>(other);
        return coulomb(*pother, Pother);
      }
      /// Return number of basis functions
      size_t nbf() const override {
        return zeta_.n_elem;
      }
    };
  }
}
