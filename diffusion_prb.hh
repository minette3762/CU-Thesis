/**
 * @brief Implements a generic transport problem for convection-diffusion simulations in PRB cases
 *
 * This class represents a flexible framework for solving convection-diffusion problems
 * with time-dependent coefficients. It supports various dimensional configurations 
 * (2D and 3D) and includes advanced features like well modeling.
 *
 * Key Features:
 * - Supports configurable grid parameters
 * - Handles well placements and characteristics
 * - Implements boundary condition logic
 * - Provides methods for tensor diffusion, velocity field, and source/sink terms
 *
 * @tparam GV Grid view type
 * @tparam VDGF Velocity discrete grid function type
 *
 * @note Utilizes Dune PDELab framework for convection-diffusion problem setup
 */

 #include"../../common/wells.hh"

 template<typename GV, typename VDGF>
 class GenericTransportProblem
 {
   typedef typename VDGF::Traits::RangeFieldType RF;
   typedef Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type BCType;
   const VDGF& vdgf;
 public:
   typedef Dune::PDELab::ConvectionDiffusionParameterTraits<GV,RF> Traits;
 
 
 private:
   typedef Wells<Traits::dimDomain,typename Traits::DomainFieldType> WellContainer;
 
   // the dimension
   enum {dim=Traits::dimDomain};
 
   //Dune::ParameterTree& ptree;
   //const YFG& yfg;
   WellContainer wells;
   std::array<RF,dim> L_, h_, grad;
   std::array<unsigned int,dim> N_;
 
 public:
 
   GenericTransportProblem (Dune::ParameterTree& ptree, const VDGF& vdgf_) :
     vdgf(vdgf_),
     wells(ptree),
     I(0.0), // zero tensor
     time(0.0)
   {
     // grid parameters
     L_ = ptree.get<std::array<RF,dim> >("grid.L");
     N_ = ptree.get<std::array<unsigned int,dim> >("grid.N");
     for (int i=0; i<dim; i++) h_[i]=L_[i]/N_[i];
 
     // diffusion coefficient
     diffusion_x = ptree.get<double>("problem.diffusion_x");
     diffusion_y = ptree.get<double>("problem.diffusion_y");
     for (std::size_t i=0; i<Traits::dimDomain; i++)
         for (std::size_t j=0; j<Traits::dimDomain; j++)
             I[i][j] = 0.0;  // Reset all elements to zero first
     // Then set diagonal elements with different values
     I[0][0] = diffusion_x;  // Value for x-direction
     I[1][1] = diffusion_y;  // Value for y-direction
 
     // initial condition
     X_ = ptree.get<double>("problem.X");
     Y_ = ptree.get<double>("problem.Y");
     H_ = ptree.get<double>("problem.H");

     decay = ptree.get<double>("problem.decay");
   }
 
   //! tensor diffusion constant per cell? return false if you want more than one evaluation of A per cell.
   static constexpr bool permeabilityIsConstantPerCell()
   {
     return true;
   }
 
   //! tensor diffusion coefficient
   typename Traits::PermTensorType
   A (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
   {
     return I;
   }
 
   //! velocity field
   typename Traits::RangeType
   b (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
   {
     typename VDGF::Traits::RangeType velo;
     vdgf.evaluate(e,x,velo);
     v[0] = velo[0];
     v[1] = velo[1];
     return v;
   }
 
   //! sink term (modified for PRB first-order decay)
  typename Traits::RangeFieldType
  c (const typename Traits::ElementType& e, const typename Traits::DomainType& x_local) const
  {
    // Define the PRB region
    double prb_x_min = 80.0; // Example PRB start x-coordinate
    double prb_x_max = 85.0; // Example PRB end x-coordinate
    // Add y and z checks if needed for 2D/3D
    double prb_y_min = 0.0;
    double prb_y_max = 100.0; // Example: Barrier spans entire y-domain

    // Define the first-order decay rate constant within the PRB (positive value)
    double decay_rate_constant_k = decay; // Example: units of 1/time (e.g., 1/day)

    // Get the global coordinates
    typename Traits::DomainType x_global = e.geometry().global(x_local);

    // Check if the point is within the PRB
    bool in_prb = false;
    if (Traits::dimDomain >= 1) // Basic check for x-dimension
    {
        if (x_global[0] >= prb_x_min && x_global[0] <= prb_x_max)
        {
            // Add further checks for y (and z in 3D) if necessary
            if (Traits::dimDomain >= 2) {
                if (x_global[1] >= prb_y_min && x_global[1] <= prb_y_max) {
            //         // If 3D, add z check here
                    in_prb = true;
                }
            // } else {
                //  in_prb = true; // Assume true if only checking x for 1D/simplicity
            }
        }
    }

    // Return the decay rate constant if inside the PRB, otherwise zero
    if (in_prb)
    {
        return decay_rate_constant_k; // Positive k makes c*u a sink term
    }
    else
    {
        return 0.0; // No reaction outside the PRB
    }
  }
 
   //! source term
   typename Traits::RangeFieldType
   f (const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
   {
     return 0.0;
   }
 
   //! boundary condition type function
   /* return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet for Dirichlet boundary conditions
    * return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann for flux boundary conditions
    * return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow for outflow boundary conditions
    */
   BCType
   bctype (const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& xlocal) const
   {
     // check velocity field
     typename Traits::DomainType iplocal = is.geometryInInside().global(xlocal);
     typename VDGF::Traits::RangeType velo;
     vdgf.evaluate(is.inside(),iplocal,velo);
     typename Traits::RangeFieldType normalvelo = velo*(is.centerUnitOuterNormal());
     if (normalvelo<=0.0)
       return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
     else
       return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow;
   }
 
   //! Dirichlet boundary condition value
   typename Traits::RangeFieldType
   g (const typename Traits::ElementType& e, const typename Traits::DomainType& xlocal) const
   {
     // 1. initial condition inside the domain
     if (time<1e-8)
     {
       typename Traits::DomainType center = e.geometry().center();
       if (Traits::dimDomain==2)
         if (center[0]>X_ && center[0]<X_+H_ && center[1]>Y_ && center[1]<Y_+H_)
           return 1.0;
       return 0.0;
     }
     return 0.0;
   }
 
   //! flux boundary condition
   typename Traits::RangeFieldType
   j (const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
   {
     return 0.0;
   }
 
   //! outflow boundary condition
   typename Traits::RangeFieldType
   o (const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
   {
     return 0.0;
   }
 
   //! set time for subsequent evaluation
   void setTime (RF t)
   {
     time = t;
   }
 
 private:
   typename Traits::PermTensorType I;
   mutable typename Traits::RangeType v;
   typename Traits::RangeFieldType X_;
   typename Traits::RangeFieldType Y_;
   typename Traits::RangeFieldType H_;
   typename Traits::RangeFieldType diffusion_x;
   typename Traits::RangeFieldType diffusion_y;
   typename Traits::RangeFieldType decay;
   RF time;
 };
 