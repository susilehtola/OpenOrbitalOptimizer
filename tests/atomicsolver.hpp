#include <openorbitaloptimizer/scfsolver.hpp>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

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
      virtual Eigen::MatrixXd overlap() const=0;
      /// Evaluate kinetic energy matrix
      virtual Eigen::MatrixXd kinetic(int l) const=0;
      /// Evaluate nuclear attraction matrix
      virtual Eigen::MatrixXd nuclear_attraction() const=0;
      /// Evaluate radial basis functions
      virtual Eigen::MatrixXd eval_f(const Eigen::VectorXd & x) const {throw std::logic_error("Not implemented!\n");};
      /// Evaluate radial basis functions' derivatives
      virtual Eigen::MatrixXd eval_df(const Eigen::VectorXd & x) const {throw std::logic_error("Not implemented!\n");};
      /// Build Coulomb matrix
      virtual Eigen::MatrixXd coulomb(const std::shared_ptr<const RadialBasis> & other, const Eigen::MatrixXd & Pother) const=0;
      /// Return number of basis functions
      virtual size_t nbf() const=0;
    };

    class STOBasis : public RadialBasis {
    private:
      /// STO exponents
      Eigen::VectorXd zeta_;
      /// STO principal quantum numbers
      Eigen::VectorXi n_;

      /// Evaluate two-electron integral
      inline double Rmnv(int m, int n, int v, double x, double y) const {
        double value = std::tgamma(m+n)/(x*y*std::pow(x+y,m+n-1))*(1+Enk(m+n-1,n-v-1,y/x)+Enk(m+n-1,m-v-1,x/y));
        //printf("Rmnv(%i,%i,%i,%e,%e) = %e\n",m,n,v,x,y,value);
        return value;
      }
      /// Pairs of basis functions
      std::vector<std::pair<size_t,size_t>> basis_function_pairs() const {
        std::vector<std::pair<size_t,size_t>> list;
        for(Eigen::Index i=0;i<zeta_.size();i++)
          for(Eigen::Index j=0;j<=i;j++)
            list.push_back(std::make_pair(static_cast<size_t>(i),static_cast<size_t>(j)));
        return list;
      }
    public:
      /// Constructor
      STOBasis(const Eigen::VectorXd & zeta, const Eigen::VectorXi & n, int angular_momentum) : RadialBasis(angular_momentum), zeta_(zeta), n_(n) {
        type_ = STOBASIS;
        if(zeta_.size() != n_.size())
          throw std::logic_error("zeta and n need to be of same size!\n");
      }
      /// Evaluate overlap matrix
      Eigen::MatrixXd overlap() const override {
        auto list(basis_function_pairs());
        Eigen::MatrixXd S(zeta_.size(), zeta_.size());
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
      Eigen::MatrixXd kinetic(int am) const override {
        auto list(basis_function_pairs());
        Eigen::MatrixXd T(zeta_.size(), zeta_.size());
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
      Eigen::MatrixXd nuclear_attraction() const override {
        auto list(basis_function_pairs());
        Eigen::MatrixXd V(zeta_.size(), zeta_.size());
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
      Eigen::MatrixXd eval_f(const Eigen::VectorXd & x) const override {
        Eigen::MatrixXd f = Eigen::MatrixXd::Zero(x.size(), zeta_.size());
#pragma omp parallel for collapse(2)
        for(Eigen::Index iz=0; iz<zeta_.size(); iz++) {
          for(Eigen::Index ix=0; ix<x.size(); ix++) {
            f(ix,iz) = std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix));
          }
        }
        return f;
      }
      /// Evaluate basis functions
      Eigen::MatrixXd eval_df(const Eigen::VectorXd & x) const override {
        Eigen::MatrixXd df = Eigen::MatrixXd::Zero(x.size(), zeta_.size());
#pragma omp parallel for collapse(2)
        for(Eigen::Index iz=0; iz<zeta_.size(); iz++) {
          for(Eigen::Index ix=0; ix<x.size(); ix++) {
            df(ix,iz) = -zeta_(iz) * std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix));
            if(n_(iz)>1) {
              df(ix,iz) += (n_(iz)-1) * std::pow(x(ix),n_(iz)-2) * std::exp(-zeta_(iz)*x(ix));
            }
          }
        }
        return df;
      }
      /// Evaluate Coulomb matrix
      Eigen::MatrixXd coulomb(const STOBasis & other, const Eigen::MatrixXd & Pother) const {
        auto list(basis_function_pairs());

        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(zeta_.size(), zeta_.size());
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          double Jij = 0.0;
          for(Eigen::Index l=0;l<Pother.cols();l++)
            for(Eigen::Index k=0;k<Pother.rows();k++)
              Jij += Pother(k,l) * Rmnv(n_(i)+n_(j), other.n_(k)+other.n_(l), 0, zeta_(i)+zeta_(j), other.zeta_(k)+other.zeta_(l));
          J(i,j) = J(j,i) = Jij;
        }
        return J;
      }
      /// Wrapper for the above
      Eigen::MatrixXd coulomb(const std::shared_ptr<const RadialBasis> & other, const Eigen::MatrixXd & Pother) const override {
        assert(other->get_type() == STOBASIS);
        auto pother = std::dynamic_pointer_cast<const STOBasis>(other);
        return coulomb(*pother, Pother);
      }
      /// Return number of basis functions
      size_t nbf() const override {
        return static_cast<size_t>(zeta_.size());
      }
    };

    class GTOBasis : public RadialBasis {
    private:
      /// GTO exponents
      Eigen::VectorXd zeta_;
      /// GTO principal quantum numbers
      Eigen::VectorXi n_;

      /// Evaluate two-electron integral
      inline double Rmnv(int m, int n, int v, double x, double y) const {
        double value = std::tgamma((m+n-1)/2.0)/(x*y*std::pow(x+y,(m+n-3)/2.0))*(1+Enk(0.5*(m+n-3),0.5*(n-v-2),y/x)+Enk(0.5*(m+n-3),0.5*(m-v-2),x/y))/4.0;
        //printf("Rmnv(%i,%i,%i,%e,%e) = %e\n",m,n,v,x,y,value);
        return value;
      }
      /// Pairs of basis functions
      std::vector<std::pair<size_t,size_t>> basis_function_pairs() const {
        std::vector<std::pair<size_t,size_t>> list;
        for(Eigen::Index i=0;i<zeta_.size();i++)
          for(Eigen::Index j=0;j<=i;j++)
            list.push_back(std::make_pair(static_cast<size_t>(i),static_cast<size_t>(j)));
        return list;
      }
    public:
      /// Constructor
      GTOBasis(const Eigen::VectorXd & zeta, const Eigen::VectorXi & n, int angular_momentum) : RadialBasis(angular_momentum), zeta_(zeta), n_(n) {
        type_ = GTOBASIS;
        if(zeta_.size() != n_.size())
          throw std::logic_error("zeta and n need to be of same size!\n");
      };
      /// Evaluate overlap matrix
      Eigen::MatrixXd overlap() const override {
        auto list(basis_function_pairs());
        Eigen::MatrixXd S(zeta_.size(), zeta_.size());
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
      Eigen::MatrixXd kinetic(int l) const override {
        auto list(basis_function_pairs());
        Eigen::MatrixXd T(zeta_.size(), zeta_.size());
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
      Eigen::MatrixXd nuclear_attraction() const override {
        auto list(basis_function_pairs());
        Eigen::MatrixXd V(zeta_.size(), zeta_.size());
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
      Eigen::MatrixXd eval_f(const Eigen::VectorXd & x) const override {
        Eigen::MatrixXd f = Eigen::MatrixXd::Zero(x.size(), zeta_.size());
#pragma omp parallel for collapse(2)
        for(Eigen::Index iz=0; iz<zeta_.size(); iz++) {
          for(Eigen::Index ix=0; ix<x.size(); ix++) {
            f(ix,iz) = std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix)*x(ix));
          }
        }
        return f;
      }
      /// Evaluate basis functions
      Eigen::MatrixXd eval_df(const Eigen::VectorXd & x) const override {
        Eigen::MatrixXd df = Eigen::MatrixXd::Zero(x.size(), zeta_.size());
#pragma omp parallel for collapse(2)
        for(Eigen::Index iz=0; iz<zeta_.size(); iz++) {
          for(Eigen::Index ix=0; ix<x.size(); ix++) {
            df(ix,iz) = -2.0 * zeta_(iz) * x(ix) * std::pow(x(ix),n_(iz)-1) * std::exp(-zeta_(iz)*x(ix)*x(ix));
            if(n_(iz)>1) {
              df(ix,iz) += (n_(iz)-1) * std::pow(x(ix),n_(iz)-2) * std::exp(-zeta_(iz)*x(ix)*x(ix));
            }
          }
        }
        return df;
      }
      /// Evaluate Coulomb matrix
      Eigen::MatrixXd coulomb(const GTOBasis & other, const Eigen::MatrixXd & Pother) const {
        auto list(basis_function_pairs());

        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(zeta_.size(), zeta_.size());
#pragma omp parallel for
        for(auto pair: list) {
          size_t i=pair.first;
          size_t j=pair.second;
          double Jij = 0.0;
          for(Eigen::Index l=0;l<Pother.cols();l++)
            for(Eigen::Index k=0;k<Pother.rows();k++)
              Jij += Pother(k,l) * Rmnv(n_(i)+n_(j), other.n_(k)+other.n_(l), 0, zeta_(i)+zeta_(j), other.zeta_(k)+other.zeta_(l));
          J(i,j) = J(j,i) = Jij;
        }
        return J;
      }
      /// Wrapper for the above
      Eigen::MatrixXd coulomb(const std::shared_ptr<const RadialBasis> & other, const Eigen::MatrixXd & Pother) const override {
        assert(other->get_type() == GTOBASIS);
        auto pother = std::dynamic_pointer_cast<const GTOBasis>(other);
        return coulomb(*pother, Pother);
      }
      /// Return number of basis functions
      size_t nbf() const override {
        return static_cast<size_t>(zeta_.size());
      }
    };
  }
}
