#include "scfsolver.hpp"
#include <armadillo>
#include <cassert>

namespace openorbital {
  namespace atomicsolver {

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
      /// Get radial basis type
      RadialBasisType get_type() const {return type_;}
      /// Coulomb integral
      virtual double Rmnv(int m, int n, int v, double x, double y) const {throw std::logic_error("Not implemented!\n");}

    public:
      RadialBasis(int angular_momentum) : angular_momentum_(angular_momentum) {};
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
        printf("Rmnv(%i,%i,%i,%e,%e) = %e\n",m,n,v,x,y,value);
        return value;
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
        arma::mat S(zeta_.n_elem, zeta_.n_elem);
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++) {
            // Pitzer, page 244
            S(i,j) = S(j,i) = Vn(n_(i)+n_(j), zeta_(i)+zeta_(j));
            printf("S(%i,%i) %e\n",i,j,S(i,j));
          }
        return S;
      };
      /// Evaluate kinetic energy matrix
      arma::mat kinetic(int am) const override {
        arma::mat T(zeta_.n_elem, zeta_.n_elem);
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++) {
            // Pitzer, page 244
            T(i,j) = T(j,i) = 0.5*zeta_(i)*zeta_(j)*(
                                                     Wn(n_(i)-am, zeta_(i)) * Wn(n_(j)-am, zeta_(j)) * Vn(n_(i)+n_(j)-2, zeta_(i)+zeta_(j))
                                                     - (Wn(n_(i)-am, zeta_(i)) + Wn(n_(j)-am, zeta_(j))) * Vn(n_(i)+n_(j)-1, zeta_(i)+zeta_(j))
                                                     + Vn(n_(i)+n_(j), zeta_(i) + zeta_(j))
                                                     );
            printf("T(%i,%i) %e\n",i,j,T(i,j));
          }
        return T;
      }
      /// Evaluate nuclear attraction matrix
      arma::mat nuclear_attraction() const override {
        arma::mat V(zeta_.n_elem, zeta_.n_elem);
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++) {
            // Pitzer, page 244
            V(i,j) = V(j,i) = Vn(n_(i)+n_(j)-1, zeta_(i)+zeta_(j));
            printf("V(%i,%i) %e\n",i,j,V(i,j));
          }
        return V;
      }
      /// Evaluate basis functions
      arma::mat eval_f(const arma::vec & x) const override {
        arma::mat f(x.n_elem, zeta_.n_elem, arma::fill::zeros);
        for(size_t iz=0; iz<zeta_.n_elem; iz++) {
          for(size_t ix=0; ix<x.n_elem; ix++) {
            f(ix,iz) = std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix));
          }
        }
        return f;
      }
      /// Evaluate Coulomb matrix
      arma::mat coulomb(const STOBasis & other, const arma::mat & Pother) const {
        arma::mat J(zeta_.n_elem, zeta_.n_elem, arma::fill::zeros);
        for(size_t j=0;j<zeta_.n_elem;j++)
          for(size_t i=0;i<zeta_.n_elem;i++) {
            double Jij = 0.0;
            for(size_t l=0;l<Pother.n_cols;l++)
              for(size_t k=0;k<Pother.n_rows;k++)
                Jij += Pother(k,l) * Rmnv(n_(i)+n_(j), other.n_(k)+other.n_(l), 0, zeta_(i)+zeta_(j), other.zeta_(k)+other.zeta_(l));
            J(i,j) = J(j,i) = Jij;
          }
        return J;
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
        printf("Rmnv(%i,%i,%i,%e,%e) = %e\n",m,n,v,x,y,value);
        return value;
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
        arma::mat S(zeta_.n_elem, zeta_.n_elem);
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++) {
            // Pitzer, page 244. Factor one half is missing in paper
            S(i,j) = S(j,i) = 0.5*Vn(0.5*(n_(i)+n_(j)-1),zeta_(i)+zeta_(j));
          }
        return S;
      };
      /// Evaluate kinetic energy matrix
      arma::mat kinetic(int l) const override {
        arma::mat T(zeta_.n_elem, zeta_.n_elem);
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++) {
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
        arma::mat V(zeta_.n_elem, zeta_.n_elem);
        for(size_t i=0;i<zeta_.n_elem;i++)
          for(size_t j=0;j<=i;j++) {
            // Pitzer, page 244. Factor one half is missing in paper
            V(i,j) = V(j,i) = 0.5*Vn(0.5*(n_(i)+n_(j)-2),zeta_(i)+zeta_(j));
          }
        return V;
      }
      /// Evaluate Coulomb matrix
      arma::mat coulomb(const GTOBasis & other, const arma::mat & Pother) const {
        arma::mat J(zeta_.n_elem, zeta_.n_elem, arma::fill::zeros);
        for(size_t j=0;j<zeta_.n_elem;j++)
          for(size_t i=0;i<zeta_.n_elem;i++) {
            double Jij = 0.0;
            for(size_t l=0;l<Pother.n_cols;l++)
              for(size_t k=0;k<Pother.n_rows;k++)
                Jij += Pother(k,l) * Rmnv(n_(i)+n_(j), other.n_(k)+other.n_(l), 0, zeta_(i)+zeta_(j), other.zeta_(k)+other.zeta_(l));
            J(i,j) = J(j,i) = Jij;
          }
        return J;
      }
    };
  }
}
