/**
 * @file main.cc
 * @brief Coupled Groundwater Flow and Contaminant Transport Solver using DG Methods.
 *
 * This file implements a fully coupled simulation for transient groundwater flow
 * and contaminant transport in heterogeneous porous media.
 *
 * @details
 * Numerical Schemes:
 * - Spatial Discretization: Discontinuous Galerkin (DG) Method (SIPG).
 * - Time Discretization: Diagonally Implicit Runge-Kutta (DIRK).
 *
 * @author Visarut Huayshelake (Adapted from Master's Thesis)
 * @date Modified: November 16, 2024
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

// =========================================================================
// DUNE & PDELab Includes
// =========================================================================
#include <dune/pdelab/boilerplate/pdelab.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh> // DG Operator
#include <dune/pdelab/localoperator/l2.hh>
#include <dune/pdelab/localoperator/darcyfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <sys/stat.h>
#include <chrono>  

// =========================================================================
// FFTW headers
// =========================================================================  
#include <fftw3.h>
#include <fftw3-mpi.h>
#include "hdf5.h"

// =========================================================================
// Project Headers
// =========================================================================
#include "df_convectiondiffusionccfv.hh"
#include "df_darcy_CCFV.hh"
#include "../../common/maxvelocity.hh"
#include "../../common/permeability_adapter.hh"
#include "../../common/limiter.hh"
#include "../../common/utilities/tools.hh"

#include "../../common/fieldgen/datatypes.hh"
#include "../../common/fieldgen/Vector.hh"
#include "../../common/fieldgen/tools.hh"
#include "../../common/fieldgen/H5Tools.hh"
#include "../../common/fieldgen/VTKPlot.hh"
#include "../../common/fieldgen/FieldData.hh"
#include "../../common/fieldgen/FFTFieldGenerator.hh"

#include "transient_GWF.hh"
#include "diffusion_prb.hh"

//***********************************************************************
// Core Simulation Function
//***********************************************************************

/**
 * @brief Executes the coupled DG simulation.
 *
 * @tparam Grid         The DUNE Grid type (e.g., YaspGrid, UGGrid).
 * @tparam degree       Polynomial degree of the basis functions.
 * @tparam elemtype     Geometry type (e.g., Cube, Simplex).
 * @tparam meshtype     Mesh type (Conforming/Non-conforming).
 * @tparam solvertype   Linear solver category (Sequential/Overlapping).
 * @tparam YFG          Type of the Random Field Generator (FFT-based).
 */
template<typename Grid, int degree, Dune::GeometryType::BasicType elemtype,
    Dune::PDELab::MeshType meshtype, Dune::SolverCategory::Category solvertype,
    typename YFG>
void do_simulation_dg (Grid& grid, std::string basename, Dune::ParameterTree& ptree, const YFG& yfg)
{
  using Dune::PDELab::Backend::native;

  // -- 1. Parameter Setup --
  typedef typename Grid::Grid GM;
  const unsigned int dim = GM::dimension;
  typedef double NumberType;

  // Extract configuration from ParameterTree (.ini file)
  bool implicit(true);
  if (ptree["simulation.method"]=="explicit") implicit=false;
  
  int order = ptree.get<int>("simulation.order");
  int every = ptree.get<int>("output.every");
  bool uselimiter = ptree.get<bool>("simulation.uselimiter");
  double theta = ptree.get<double>("simulation.theta");
  NumberType hx, hy;
  NumberType Cr = ptree.get<double>("simulation.Courant");

  //===============================================================
  // PART A: FLOW PROBLEM SOLVER
  //===============================================================
  
  // Define the flow problem
  typedef GenericTransientFlowProblem<typename GM::LeafGridView,NumberType,YFG> ProblemF;
  ProblemF problemf(ptree,yfg);

  // BC Adapter
  typedef Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<ProblemF> BCTypeF;
  BCTypeF bctypef(grid->leafGridView(),problemf);

  // Construct Finite Element Space for Flow
  typedef Dune::PDELab::DGQkSpace<GM,NumberType,degree,elemtype,solvertype> FSF;
  FSF fsf(grid->leafGridView());

  // Degree of Freedom Vector Setup
  typedef typename FSF::DOF VF;
  VF xf(fsf.getGFS(), 0.0); // Solution vector for Head
  
  // Interpolate Initial Conditions
  typename FSF::DGF xfdgf(fsf.getGFS(),xf);
  typedef Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter<ProblemF> GF;
  GF gf(grid->leafGridView(),problemf);
  problemf.setTime(0.0);
  Dune::PDELab::interpolate(gf,fsf.getGFS(),xf);

  // Assembler Setup
  const std::size_t nonzeros = 2*dim+1;

  // Local Operator: ConvectionDiffusionDG
  typedef Dune::PDELab::ConvectionDiffusionDG<ProblemF, typename FSF::FEM> LOPF;
  LOPF lopf(problemf, 
            Dune::PDELab::ConvectionDiffusionDGMethod::SIPG,
            Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn,
            2.0); // Penalty parameter

  // Global Assemblers
  typedef Dune::PDELab::GalerkinGlobalAssembler<FSF,LOPF,solvertype> SASSF;
  SASSF sassf(fsf,lopf,nonzeros);
  
  typedef Dune::PDELab::L2 MLOPF;
  MLOPF mlopf(2*degree);
  typedef Dune::PDELab::GalerkinGlobalAssembler<FSF,MLOPF,solvertype> TASSF;
  TASSF tassf(fsf,mlopf,1);

  // Time-stepping Assemblers
  typedef Dune::PDELab::OneStepGlobalAssembler<SASSF,TASSF,true> ASSEMBLERIF;
  ASSEMBLERIF assemblerif(sassf,tassf);
  
  typedef Dune::PDELab::OneStepGlobalAssembler<SASSF,TASSF,false> ASSEMBLEREF;
  ASSEMBLEREF assembleref(sassf,tassf);

  // Iterative Solver
  typedef Dune::PDELab::ISTLSolverBackend_IterativeDefault<FSF,ASSEMBLERIF,solvertype> SBEIF;
  SBEIF sbeif(fsf,assemblerif,5000,1);
  
  // Explicit Diagonal Solver
  typedef Dune::PDELab::ISTLSolverBackend_ExplicitDiagonal<FSF,ASSEMBLEREF,solvertype> SBEEF;
  SBEEF sbeef(fsf,assembleref,5000,1);

  // Conjugate Gradient with SSOR Preconditioner
  typedef Dune::PDELab::ISTLSolverBackend_CG_SSOR <FSF,ASSEMBLERIF,solvertype> SBEF;
  SBEF sbef(fsf,assemblerif,5000,1,1);

  typedef Dune::PDELab::StationaryLinearProblemSolver<typename ASSEMBLERIF::GO,typename SBEF::LS,VF> PDESOLVERIF;
  PDESOLVERIF pdesolverif(*assemblerif,*sbef,xf,1e-8);
  std::cout << "Using CGCS Linear Solver for Flow" << std::endl;

  typedef Dune::PDELab::StationaryLinearProblemSolver<typename ASSEMBLEREF::GO,typename SBEEF::LS,VF> PDESOLVEREF;
  PDESOLVEREF pdesolveref(*assembleref,*sbeef,1e-8);

  // Time Stepper Configuration
  Dune::PDELab::OneStepThetaParameter<NumberType> methodif1(0.5); // Crank-Nicolson
  Dune::PDELab::Alexander2Parameter<NumberType> methodif2;
  Dune::PDELab::Alexander3Parameter<NumberType> methodif3; // DIRK3
  
  Dune::PDELab::TimeSteppingParameterInterface<NumberType>* pmethodif;
  if (order==1) pmethodif = &methodif1;
  if (order==2) pmethodif = &methodif2;
  if (order==3) pmethodif = &methodif3;

  typedef Dune::PDELab::OneStepMethod<NumberType,typename ASSEMBLERIF::GO,PDESOLVERIF,VF> OSMIF;
  OSMIF osmif(*pmethodif,*assemblerif,pdesolverif);
  osmif.setVerbosityLevel(2);

  // Explicit methods setup
  Dune::PDELab::ExplicitEulerParameter<NumberType> methodef1;
  Dune::PDELab::HeunParameter<NumberType> methodef2;
  Dune::PDELab::Shu3Parameter<NumberType> methodef3;
  
  typedef Dune::PDELab::SimpleTimeController<NumberType> TCF;
  TCF tcf;
  
  Dune::PDELab::TimeSteppingParameterInterface<NumberType>* pmethodef;
  if (order==1) pmethodef = &methodef1;
  if (order==2) pmethodef = &methodef2;
  if (order==3) pmethodef = &methodef3;
  
  typedef Dune::PDELab::ExplicitOneStepMethod<NumberType,typename ASSEMBLEREF::GO,typename SBEEF::LS,VF,VF,TCF> OSMEF;
  OSMEF osmef(*pmethodef,*assembleref,*sbeef,tcf);
  osmef.setVerbosityLevel(2);

  // lope Limiter Setup
  Dune::PDELab::Limiter<typename FSF::GFS> limiter(fsf.getGFS(),theta);

  //===============================================================
  // PART B: VELOCITY & PERMEABILITY FIELD
  //===============================================================
  
  // Export Permeability Field to Vector
  Vector<NumberType> log_permeability_field; 
  yfg.export_field_to_vector_on_grid(grid->leafGridView(), log_permeability_field);

  // Map to P0 Grid Function (Cell-centered)
  // NOTE: This assumes structured cubic grid instantiation.
  typedef Dune::PDELab::P0LocalFiniteElementMap<typename GM::ctype,NumberType,dim> P0FEM;
  typedef Dune::PDELab::NoConstraints NOCONS;
  typedef Dune::PDELab::ISTL::VectorBackend<> VBE;
  
  P0FEM p0fem(Dune::GeometryTypes::cube(dim));
  typedef Dune::PDELab::GridFunctionSpace<typename GM::LeafGridView,P0FEM,NOCONS,VBE> P0GFS;
  P0GFS p0gfs(grid->leafGridView(),p0fem);
  p0gfs.name("Y");
  
  using P0VCType = Dune::PDELab::Backend::Vector<P0GFS,NumberType>;
  P0VCType p0Celldata(p0gfs,0.0);
  
  // Copy data to P0 vector
  for(unsigned int i = 0; i < log_permeability_field.size(); ++i)
    native(p0Celldata)[i] = log_permeability_field[i];

  // Compute Darcy Velocity
  typedef DarcyVelocityFromHeadFEM<ProblemF, typename FSF::GFS, VF> DarcyDGF;
  DarcyDGF darcydgf(problemf,fsf.getGFS(),xf);
  
  typedef Dune::PDELab::DiscreteGridFunction<P0GFS,P0VCType> P0DGF;
  P0DGF p0dgf(p0gfs,p0Celldata);

  //===============================================================
  // PART C: TRANSPORT PROBLEM SOLVER
  //===============================================================

  // Define Transport Problem
  typedef GenericTransportProblem<typename GM::LeafGridView,DarcyDGF> Problem;
  Problem problem(ptree,darcydgf);
  
  typedef Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<Problem> BCType;
  BCType bctype(grid->leafGridView(),problem);

  // Finite Element Space for Transport
  typedef Dune::PDELab::DGQkSpace<GM,NumberType,degree,elemtype,solvertype> FS;
  FS fs(grid->leafGridView());

  // Assemblers for Transport (DG)
  typedef Dune::PDELab::ConvectionDiffusionDG<Problem,typename FS::FEM> LOP;
  LOP lop(problem,Dune::PDELab::ConvectionDiffusionDGMethod::SIPG,Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn,2.0);
  
  typedef Dune::PDELab::GalerkinGlobalAssembler<FS,LOP,solvertype> SASS;
  SASS sass(fs,lop,nonzeros);
  
  typedef Dune::PDELab::L2 MLOP;
  MLOP mlop(2*degree);
  typedef Dune::PDELab::GalerkinGlobalAssembler<FS,MLOP,solvertype> TASS;
  TASS tass(fs,mlop,1);
  
  typedef Dune::PDELab::OneStepGlobalAssembler<SASS,TASS,true> ASSEMBLERI;
  ASSEMBLERI assembleri(sass,tass);
  typedef Dune::PDELab::OneStepGlobalAssembler<SASS,TASS,false> ASSEMBLERE;
  ASSEMBLERE assemblere(sass,tass);

  // Initial Value for Transport
  typedef typename FS::DOF V;
  V x(fs.getGFS(),0.0);
  typedef Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter<Problem> G;
  G g(grid->leafGridView(),problem);
  problem.setTime(0.0);
  Dune::PDELab::interpolate(g,fs.getGFS(),x);

  // Linear Solvers for Transport
  typedef Dune::PDELab::ISTLSolverBackend_IterativeDefault<FS,ASSEMBLERI,solvertype> SBEI;
  SBEI sbei(fs,assembleri,5000,1);
  typedef Dune::PDELab::ISTLSolverBackend_ExplicitDiagonal<FS,ASSEMBLERE,solvertype> SBEE;
  SBEE sbee(fs,assemblere,5000,1);

  // Using CG with SSOR
  typedef Dune::PDELab::ISTLSolverBackend_CG_SSOR <FS,ASSEMBLERI,solvertype> SBE;
  SBE sbe(fs,assembleri,5000,1,1);
  
  typedef Dune::PDELab::StationaryLinearProblemSolver<typename ASSEMBLERI::GO,typename SBE::LS,V> PDESOLVERI;
  PDESOLVERI pdesolveri(*assembleri,*sbe,x,1e-8);

  typedef Dune::PDELab::StationaryLinearProblemSolver<typename ASSEMBLERE::GO,typename SBEE::LS,V> PDESOLVERE;
  PDESOLVERE pdesolvere(*assemblere,*sbee,1e-8);

  // Time Steppers for Transport
  Dune::PDELab::OneStepThetaParameter<NumberType> methodi1(0.5);
  Dune::PDELab::Alexander2Parameter<NumberType> methodi2;
  Dune::PDELab::Alexander3Parameter<NumberType> methodi3;
  
  Dune::PDELab::TimeSteppingParameterInterface<NumberType>* pmethodi;
  if (order==1) pmethodi = &methodi1;
  if (order==2) pmethodi = &methodi2;
  if (order==3) pmethodi = &methodi3;
  
  typedef Dune::PDELab::OneStepMethod<NumberType,typename ASSEMBLERI::GO,PDESOLVERI,V> OSMI;
  OSMI osmi(*pmethodi,*assembleri,pdesolveri);
  osmi.setVerbosityLevel(2);

  // Explicit Transport
  Dune::PDELab::ExplicitEulerParameter<NumberType> methode1;
  Dune::PDELab::HeunParameter<NumberType> methode2;
  Dune::PDELab::Shu3Parameter<NumberType> methode3;
  
  typedef Dune::PDELab::SimpleTimeController<NumberType> TC;
  TC tc;
  
  Dune::PDELab::TimeSteppingParameterInterface<NumberType>* pmethode;
  if (order==1) pmethode = &methode1;
  if (order==2) pmethode = &methode2;
  if (order==3) pmethode = &methode3;
  
  typedef Dune::PDELab::ExplicitOneStepMethod<NumberType,typename ASSEMBLERE::GO,typename SBEE::LS,V,V,TC> OSME;
  OSME osme(*pmethode,*assemblere,*sbee,tc);
  osme.setVerbosityLevel(2);

  //===============================================================
  // PART D: TIME LOOP
  //===============================================================
  
  // Grid parameters for CFL Calculation
  auto L = ptree.get<std::array<double,dim> >("grid.L");
  auto N = ptree.get<std::array<unsigned int,dim> >("grid.N");
  double dt = ptree.get<double> ("simulation.dt");
  double steps = ptree.get<double>("simulation.steps");
  hx = L[0]/N[0];
  hy = L[1]/N[1];
  NumberType h = hx; h = std::min(h, hy);
  NumberType maxv = maxvelocity(darcydgf);
  
  std::cout << "Initialization info:" << std::endl;
  std::cout << "  Maximum velocity = " << maxv << " m/s" << std::endl;
  std::cout << "  Grid size h = " << h << " m" << std::endl;
  
  NumberType dtMo = Cr*h/(((double)dim)*maxv);
  std::cout << "  Simulation dt = " << dt << " s (Suggested dt < " << dtMo << ")" << std::endl;

  NumberType time = 0.0;
  int step = 1;

  if (uselimiter) limiter.poststage(x);
  
  // Output Setup
  Dune::FunCEP::createDirectory(basename);
  auto stationaryvtkwriter = std::make_shared<Dune::SubsamplingVTKWriter<typename GM::LeafGridView>>(grid->leafGridView(),Dune::refinementLevels(std::max(0,degree-1)));
  Dune::VTKSequenceWriter<typename GM::LeafGridView> vtkwriter(stationaryvtkwriter,basename,basename,"");
  
  // Register Data Fields
  vtkwriter.addVertexData(std::make_shared<typename FSF::VTKF>(xfdgf,"Head")); // Head
  typename FS::DGF xdgf(fs.getGFS(),x);
  vtkwriter.addVertexData(std::make_shared<typename FS::VTKF>(xdgf,"Concentration"));     // Concentration
  typedef Dune::PDELab::VTKGridFunctionAdapter<DarcyDGF> DarcyVTKDGF;
  vtkwriter.addVertexData(std::make_shared<DarcyVTKDGF>(darcydgf,"Velocity"));       // Velocity
  typedef Dune::PDELab::VTKGridFunctionAdapter<P0DGF> VTKP0;
  vtkwriter.addCellData(std::make_shared<VTKP0>(p0dgf,"Y"));                    // Permeability (Log)
  typedef DiagonalPermeabilityAdapter<ProblemF> PermDGF;
  PermDGF permdgf(grid->leafGridView(),problemf);
  typedef Dune::PDELab::VTKGridFunctionAdapter<PermDGF> PermVTKDGF;
  vtkwriter.addCellData(std::make_shared<PermVTKDGF>(permdgf,"K"));             // Conductivity
  
  // Write Initial State
  vtkwriter.write(time,Dune::VTK::appendedraw);

  // Start Time Loop
  while (step <= steps)
  {
      // 1. Solve Flow (Updated every step if transient/coupled)
      VF xnewf(fsf.getGFS(),0.0);
      if(implicit)
        osmif.apply(time,dt,xf,gf,xnewf);
      else {
        if (uselimiter)
          osme.apply(time,dt,xf,xnewf,limiter);
        else
        osme.apply(time,dt,xf,xnewf);
      }
      xf = xnewf;

      // 2. Solve Transport
      V xnew(fs.getGFS(),0.0);
      if (implicit)
        osmi.apply(time,dt,x,xnew);
      else {
        if (uselimiter)
          osme.apply(time,dt,x,xnew,limiter);
        else
        osme.apply(time,dt,x,xnew);
      }

      // 3. Update State
      x = xnew;
      time += dt;
      step++;

      // 4. Output
      if (step % every == 0) {
        vtkwriter.write(time,Dune::VTK::appendedraw);
        std::cout << "[Step " << step << "] Time: " << time << " | Max Velocity: " << maxv << std::endl;
      }
  }
  
  std::cout << "Simulation Completed." << std::endl;
}

//***********************************************************************
// Main Part
//***********************************************************************

int main(int argc, char **argv)
{
  // Initialize MPI
  Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);

  // Parse Parameters
  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("couple.ini",ptree);
  ptreeparser.readOptions(argc,argv,ptree);

  // Solver Constants
  constexpr Dune::SolverCategory::Category solvertype = Dune::SolverCategory::overlapping;
  constexpr Dune::PDELab::MeshType meshtype = Dune::PDELab::MeshType::conforming;
  std::string basename = ptree.get<std::string>("output.basename");
  int order= ptree.get<int>("simulation.order");

  try {
    if (ptree.get("grid.dim",(int)2)==2)
    {
        auto start = std::chrono::high_resolution_clock::now();
        constexpr unsigned int dim=2;

        // Grid Generation (Structured)
        if (ptree["grid.type"]=="structured")
        {
          std::array<double,dim> lower_left; 
          for (unsigned int i=0; i<dim; i++) lower_left[i]=0.0;
          
          auto upper_right = ptree.get<std::array<double,dim> >("grid.L");
          auto cells = ptree.get<std::array<unsigned int,dim> >("grid.N");

          if (ptree["grid.manager"]=="yasp")
          {
            // Create YaspGrid
            constexpr Dune::GeometryType::BasicType elemtype = Dune::GeometryType::cube;
            typedef Dune::YaspGrid<dim> GM;
            typedef Dune::PDELab::StructuredGrid<GM> Grid;
            Grid grid(elemtype,lower_left,upper_right,cells);
            grid->globalRefine(ptree.get("grid.refinement",(int)0));

            // Setup Heterogeneous Field Generator
            std::cout << "Setting up random field generator..." << std::endl;
            auto fielddir = ptree.get<std::string>("problem.field");
            Dune::ParameterTree fieldptree;
            std::string fieldinifile = fielddir + "/param.ini";
            ptreeparser.readINITree(fieldinifile,fieldptree);
            
            typedef Dune::FunCEP::FieldData<double> FD;
            FD fielddata(dim);
            fielddata.read(fieldptree);
            
            if(helper.rank() == 0)
              fielddata.printInfos();
            
            typedef Dune::FunCEP::FFTFieldGenerator<FD,REAL,dim> YFG;
            YFG yfg(fielddata, helper.getCommunicator());

            // Read Field Data
            if(helper.size() > 1) {
              yfg.h5g_pRead(grid->leafGridView(), fielddir + "/YField.h5", "YField");
            } else {
              yfg.h5_Read(fielddir + "/YField.h5", "YField");
            }

            // Construct output filename based on configuration
            bool implicit(true);
            if (ptree["simulation.method"]=="explicit") implicit=false;
            bool uselimiter = ptree.get<bool>("simulation.uselimiter");
            
            std::string lim("nolimiter");
            if(uselimiter) lim = "withlimiter";
            
            std::stringstream fullbasename;
            if(order == 1 || implicit) {
              fullbasename << basename << "_" << ptree["simulation.method"] << "_dim" << dim;
            } else {
              fullbasename << basename << "_" << ptree["simulation.method"] << "_" << lim << "_dim" << dim;
            }
            
            // Launch Simulation based on order
            if (order==1) {
              fullbasename << "_order" << 1;
              do_simulation_dg<Grid,1,elemtype,meshtype,solvertype>(grid,fullbasename.str(),ptree,yfg);
            }
            if (order==2) {
              fullbasename << "_order" << 2;
              do_simulation_dg<Grid,1,elemtype,meshtype,solvertype>(grid,fullbasename.str(),ptree,yfg);
            }
            if (order==3) {
              fullbasename << "_order" << 3;
              do_simulation_dg<Grid,2,elemtype,meshtype,solvertype>(grid,fullbasename.str(),ptree,yfg);
            }
          }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds> (stop-start);
        std::cout << "Total execution time: " << duration.count() << " seconds." << std::endl;
    }
  }
  catch (Dune::Exception & e) {
    std::cerr << "DUNE ERROR: " << e.what() << std::endl;
    return 1;
  }
  catch (std::exception & e) {
    std::cerr << "STL ERROR: " << e.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "Unknown ERROR" << std::endl;
    return 1;
  }

  return 0;
}