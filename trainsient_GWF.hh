/** Parameter class for the time dependent convection-diffusion equation of the following form:
 *
 * \f{align*}{
 *   \nabla\cdot(-A(x) \nabla u + b(x) u) + c(x)u &=& f \mbox{ in } \Omega,  \ \
 *                                              u &=& g \mbox{ on } \partial\Omega_D (Dirichlet)\ \
 *                (b(x,u) - A(x)\nabla u) \cdot n &=& j \mbox{ on } \partial\Omega_N (Flux)\ \
 *                        -(A(x)\nabla u) \cdot n &=& o \mbox{ on } \partial\Omega_O (Outflow)
 * \f}
 * Note:
 *  - This formulation is valid for velocity fields which are non-divergence free.
 *  - Outflow boundary conditions should only be set on the outflow boundary
 *
 * The template parameters are:
 *  - GV a model of a GridView
 *  - RF numeric type to represent results
 */

#include"../../common/wells.hh"

template<typename GV, typename RF, typename YFG>
class GenericTransientFlowProblem
{

public:
  // export traits
  typedef Dune::PDELab::ConvectionDiffusionParameterTraits<GV,RF> Traits;

private:
  // locally used types
  typedef Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type BCType;
  typedef Wells<Traits::dimDomain,typename Traits::DomainFieldType> WellContainer;

  // the dimension
  enum {dim=Traits::dimDomain};

  Dune::ParameterTree& ptree;
  const YFG& yfg;
  WellContainer wells;
  std::array<RF,dim> L, h, head;
  std::array<unsigned int,dim> N;
  // auto init;

public:

  GenericTransientFlowProblem (Dune::ParameterTree& ptree_, const YFG& yfg_) :
    ptree(ptree_),
    yfg(yfg_),
    wells(ptree_)
  {
    // read structured grid parameters
    L = ptree.get<std::array<RF,dim> >("grid.L");
    N = ptree.get<std::array<unsigned int,dim> >("grid.N");
    for (int i=0; i<dim; i++) h[i]=L[i]/N[i];

    // read regional gradient parameter
    head = ptree.get<std::array<RF,dim> >("problem.head");
    // Porosity


  } // end constructor

  //! tensor diffusion constant per cell? return false if you want more than one evaluation of A per cell.
  static constexpr bool permeabilityIsConstantPerCell()
  {
    return true;
  }

  //! tensor diffusion coefficient
  typename Traits::PermTensorType
  A (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    // evaluate cell geometry
    auto geo =  e.geometry();
    auto xglobal = geo.global(x);

    // initialize isotropic tensor from field
    double k;
    yfg.evaluateK(xglobal,k);
    typename Traits::PermTensorType K(0.0);
    for(std::size_t i=0; i<Traits::dimDomain; i++)
      K[i][i] = k;
    // modify for resolved wells
    for (int i=0; i<wells.size(); i++)
      if (wells.type(i)!=WellContainer::Point && wells.cell_in_well(i,geo))
        {
          if (wells.type(i)==WellContainer::Vertical)
            K[dim-1][dim-1]=wells[i].permeability;
          else if (wells.type(i)==WellContainer::Region)
            for (int j=0; j<dim; j++) K[j][j]=wells[i].permeability;
        }
    return K;
  }

  //! velocity field
  typename Traits::RangeType
  b (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::RangeType v(0.0);
    return v;
  }

  //! sink term
  typename Traits::RangeFieldType
  c (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    return 0.0;
  }

  //! source term
  typename Traits::RangeFieldType
  f (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    // wells, new style
    auto geo =  e.geometry();
    for (int i=0; i<wells.size(); i++)
      {
        if (wells.type(i)==WellContainer::Point || wells.type(i)==WellContainer::Vertical)
          if (wells.delta_in_cell(i,geo))
            return wells[i].rate/geo.volume();
        if (wells.type(i)==WellContainer::Region)
          if (wells.cell_in_well(i,geo))
            return wells[i].rate/wells[i].volume;
      }
    return 0.0;
  }

  //! boundary condition type function
  BCType
  bctype (const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    auto xglobal = is.geometry().global(x);
    if (xglobal[0]<1e-6) return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    if (xglobal[0]>L[0]-1e-6) return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }

  //! Dirichlet boundary condition value
  typename Traits::RangeFieldType
  g (const typename Traits::ElementType& e, const typename Traits::DomainType& xlocal) const
  {
    // boundary condition for time > 0
    auto xglobal = e.geometry().global(xlocal);
    if(xglobal[0]<1e-6)
      return head[0];
    return head[1];
  }

  //! flux boundary condition
  typename Traits::RangeFieldType
  j (const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    // in 2D the wells are implemented as source/sink
    if (Traits::dimDomain==2) return 0.0;

    // in 3D its a boundary condition term
    return 0.0;
  }

  //! outflow boundary condition
  typename Traits::RangeFieldType
  o (const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    return 0.0;
  }

  //! set time for subsequent evaluation
  void setTime(RF t)
  {
    time = t;
  }
  
private:
  RF time;
};