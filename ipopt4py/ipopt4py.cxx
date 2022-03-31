/* 2021 - 2022 Jan Provaznik (provaznik@optics.upol.cz)
 *
 * Interfaces COIN-OR IPOPT.
 * Unlike the competition it tries to keep you sane.
 *
 * Without any documentation.
 * Alright. Maybe not as sane as it could.
 *
 * Will be,
 * at some point of time and space,
 * making more sense.
 */

#include <limits>
#include <string>
#include <algorithm>

#include <iostream>
#include <boost/format.hpp>

// Boost.Python support

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace bpy = boost::python;
namespace bnp = boost::python::numpy;

// COIN-OR IPOPT support

#include <coin/IpTNLP.hpp>
#include <coin/IpIpoptApplication.hpp>

namespace coin = Ipopt;

using coin_style_t = coin::TNLP::IndexStyleEnum;
using coin_index_t = coin::Index;
using coin_value_t = coin::Number;
using coin_state_t = coin::SolverReturn;

//

template <typename T>
bnp::ndarray 
make_ndarray_from_data (
  size_t size,
  const T * const data
);

template <typename T>
size_t 
copy_ndarray_into_data (
  size_t size,
  T * const data,
  const bnp::ndarray & array
);

// Warning! 
// The make_ndarray procedure does NOT make a copy of the data.

template <typename T>
bnp::ndarray 
make_ndarray_from_data (
  size_t          size,
  const T * const data
) {

  return bnp::from_data(
    data,
    bnp::dtype::get_builtin<T>(),
    bpy::make_tuple(size),
    bpy::make_tuple(sizeof(T)),
    bpy::object()
  );
}

//

template <typename T>
size_t 
copy_ndarray_into_data (
  size_t size,
  T * const data,
  const bnp::ndarray & array
) {

  /* Make sure the underlying types are compatible.
   */

  if (array.get_dtype() != bnp::dtype::get_builtin<T>()) {
    throw std::runtime_error("The numpy.ndarray can not be converted to the desired C-type.");
  }

  // Make sure the array is contiguous: memcpy would not work.

  if (! (array.get_flags() & bnp::ndarray::CARRAY_RO)) {
    throw std::runtime_error("The numpy.ndarray is not C-ordered.");
  }

  size_t ndarray_size = 1;
  for (size_t index = 0, ndims = array.get_nd(); index < ndims; ++index) {
    ndarray_size = ndarray_size * array.shape(index);
  }

  size_t count = std::min(ndarray_size, size);
  size_t bytes = count * sizeof(T);

  /* We copy on byte level.
   * This may or may not be smart.
   */

  std::memcpy(data, array.get_data(), bytes);

  /* Return the number of elements we actually copied.
   */

  return count;
}

//

struct proxy_nlp_result {
  int status;
  int success;
  std::string message;
  bpy::object fval;
  bpy::object xval;
  bpy::object gval;
};

/* Proxied non-linear problem. */

struct proxy_nlp : coin::TNLP { 

  proxy_nlp (
    bpy::object evalf, bpy::object gradf,
    bpy::object evalg, bpy::object gradg,
    bpy::object xstart,
    bpy::object xcount, bpy::object xlimlo, bpy::object xlimhi, 
    bpy::object gcount, bpy::object glimlo, bpy::object glimhi
  ) : $evalf(evalf), $gradf(gradf),
      $evalg(evalg), $gradg(gradg), 
      $xstart(xstart), 
      $xcount(xcount), $xlimlo(xlimlo), $xlimhi(xlimhi),
      $gcount(gcount), $glimlo(glimlo), $glimhi(glimhi) {
  }

  virtual ~ proxy_nlp () {
  }

  /* Updates the currently probed x-value. */

  void update (
    coin_index_t xlen,
    const coin_value_t * const xval
  ) {
    $xpoint = make_ndarray_from_data(xlen, xval).copy();
  }

  // Construct sparse indices for a dense jacobian of constraints in C-order.

  void
  setindices (
    coin_index_t   xlen,
    coin_index_t   glen,
    coin_index_t * jrow,
    coin_index_t * jcol
  ) const {
    for (coin_index_t row = 0, off = 0; row < glen; ++row) {
      for (coin_index_t col = 0; col < xlen; ++col, ++off) {
        jrow[off] = row;
        jcol[off] = col;
      }
    }
  }

  // Information about the problem

  virtual bool 
  get_nlp_info (
    coin_index_t & xlen,
    coin_index_t & glen,
    coin_index_t & jlen,
    coin_index_t & hlen,
    coin_style_t & flag
  ) {

    xlen = boost::python::extract<coin_index_t>($xcount);
    glen = boost::python::extract<coin_index_t>($gcount);

    // Assume dense jacobian of constraints.
    jlen = glen * xlen;

    // Assume dense hessian of lagrangian.
    hlen = xlen * xlen;

    // Enforce 0-based indexing of sparse matrices.
    flag = coin::TNLP::C_STYLE;

    return true;
  }

  // 

  virtual bool 
  get_bounds_info (
    coin_index_t   xlen,
    coin_value_t * xmin,
    coin_value_t * xmax,
    coin_index_t   glen,
    coin_value_t * gmin,
    coin_value_t * gmax
  ) {

    size_t count;

    // xlimlo, xlimhi

    count = copy_ndarray_into_data(xlen, xmin, make_ndarray($xlimlo));
    if (xlen != count) {
      throw std::runtime_error("Lower limit (xlimlo) in get_bounds_info shorter than xcount.");
    }

    count = copy_ndarray_into_data(xlen, xmax, make_ndarray($xlimhi));
    if (xlen != count) {
      throw std::runtime_error("Upper limit (xlimhi) in get_bounds_info shorter than xcount.");
    }

    // glimlo, glimhi

    count = copy_ndarray_into_data(glen, gmin, make_ndarray($glimlo));
    if (glen != count) {
      throw std::runtime_error("Upper limit (glimlo) in get_bounds_info shorter than gcount.");
    }

    count = copy_ndarray_into_data(glen, gmax, make_ndarray($glimhi));
    if (glen != count) {
      throw std::runtime_error("Upper limit (glimhi) in get_bounds_info shorter than gcount.");
    }

    return true;
  }

  //
  
  virtual bool 
  get_starting_point (
    coin_index_t xlen,
    bool xset, coin_value_t * xval,
    bool zset, coin_value_t * zmin, coin_value_t * zmax,
    coin_index_t glen,
    bool lset, coin_value_t * lval
  ) {

    size_t count;

    if (zset || lset) {
      throw std::runtime_error("Setting (bound) multipliers not supported in get_starting_point.");
    }

    // xstart

    if (xset) {
      count = copy_ndarray_into_data(xlen, xval, make_ndarray($xstart));
      if (xlen != count) {
        throw std::runtime_error("Starting point (xstart) in get_starting_point shorter than xcount.");
      }
    }

    return true;
  }

  //
  
  virtual bool 
  eval_f (
    coin_index_t xlen, const coin_value_t * xval, bool xnew,
    coin_value_t & fval
  ) {

    if (xnew) {
      update(xlen, xval);
    }

    bpy::object retval = $evalf(xnew, $xpoint);
    bpy::extract<coin_value_t> result(retval);

    if (! result.check()) {
      throw std::runtime_error("The result of evalf can not be converted to the desired C-type.");
    }

    fval = result();

    return true;
  }

  //

  virtual bool 
  eval_g (
    coin_index_t xlen, const coin_value_t * xval, bool xnew,
    coin_index_t glen, coin_value_t * gval
  ) {

    size_t count;

    if (xnew) {
      update(xlen, xval);
    }

    bpy::object retval = $evalg(xnew, $xpoint);
    bnp::ndarray result = make_ndarray(retval);

    count = copy_ndarray_into_data(glen, gval, result);
    if (glen != count) {
      throw std::runtime_error("The result of evalg is shorter than gcount.");
    }

    return true;
  }

  //

  virtual bool 
  eval_grad_f (
    coin_index_t xlen, const coin_value_t * xval, bool xnew,
    coin_value_t * jval
  ) {

    size_t count;

    if (xnew) {
      update(xlen, xval);
    }

    bpy::object retval = $gradf(xnew, $xpoint);
    bnp::ndarray result = make_ndarray(retval);

    count = copy_ndarray_into_data(xlen, jval, result);
    if (xlen != count) {
      throw std::runtime_error("The result of gradf is shorter than xcount.");
    }

    return true;
  }

  // 

  virtual bool 
  eval_jac_g (
    coin_index_t xlen, const coin_value_t * xval, bool xnew,
    coin_index_t glen,
    coin_index_t jlen,
    coin_index_t * jrow, coin_index_t * jcol, coin_value_t * jval
  ) {

    size_t count;

    // The first call to eval_jac_g should set up the sparsity structure.
    // The first call has NULL xval and jval arguments.

    if (! xval && ! jval) {
      setindices(xlen, glen, jrow, jcol);
      return true;
    }

    if (xnew) {
      update(xlen, xval);
    }

    bpy::object retval = $gradg(xnew, $xpoint);
    bnp::ndarray result = make_ndarray(retval);

    count = copy_ndarray_into_data(jlen, jval, result);
    if (jlen != count) {
      throw std::runtime_error("The result of gradg is shorter than xcount * gcount.");
    }

    return true;
  }

  //

  virtual void finalize_solution (
    coin_state_t status,
    coin_index_t xlen, const coin_value_t * xval,
    const coin_value_t *, 
    const coin_value_t *,
    coin_index_t glen, const coin_value_t * gval,
    const coin_value_t *,
    coin_value_t fval,
    const coin::IpoptData *, 
    coin::IpoptCalculatedQuantities *
  ) {
    $result.status  = status;
    $result.message = message(status);
    $result.success = (status == coin::SUCCESS);

    $result.fval = bpy::object(fval);
    $result.xval = make_ndarray_from_data(xlen, xval).copy();
    $result.gval = make_ndarray_from_data(glen, gval).copy();
  }

  //

  std::string message (int status) {
    switch (status) {
      case coin::SUCCESS:
        return "SUCCESS";

      case coin::MAXITER_EXCEEDED:
        return "MAXITER_EXCEEDED";

      case coin::CPUTIME_EXCEEDED:
        return "CPUTIME_EXCEEDED";

      case coin::STOP_AT_TINY_STEP:
        return "STOP_AT_TINY_STEP";

      case coin::STOP_AT_ACCEPTABLE_POINT:
        return "STOP_AT_ACCEPTABLE_POINT";

      case coin::LOCAL_INFEASIBILITY:
        return "LOCAL_INFEASIBILITY";

      case coin::USER_REQUESTED_STOP:
        return "USER_REQUESTED_STOP";

      case coin::FEASIBLE_POINT_FOUND:
        return "FEASIBLE_POINT_FOUND";

      case coin::DIVERGING_ITERATES:
        return "DIVERGING_ITERATES";

      case coin::RESTORATION_FAILURE:
        return "RESTORATION_FAILURE";

      case coin::ERROR_IN_STEP_COMPUTATION:
        return "ERROR_IN_STEP_COMPUTATION";

      case coin::INVALID_NUMBER_DETECTED:
        return "INVALID_NUMBER_DETECTED";

      case coin::TOO_FEW_DEGREES_OF_FREEDOM:
        return "TOO_FEW_DEGREES_OF_FREEDOM";

      case coin::INVALID_OPTION:
        return "INVALID_OPTION";

      case coin::OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";

      case coin::INTERNAL_ERROR:
        return "INTERNAL_ERROR";

      case coin::UNASSIGNED:
      default:
        return "UNASSIGNED";
    }
  }

  inline 
  bnp::ndarray 
  make_ndarray (
    bpy::object from
  ) const{
    return bnp::array(from, $dtype);
  }

  public:
    proxy_nlp_result $result;

  private:
    bnp::dtype $dtype = bnp::dtype::get_builtin<coin_value_t>();
    bpy::object $xpoint;

  private:
    bpy::object $evalf; bpy::object $gradf;
    bpy::object $evalg; bpy::object $gradg;
    bpy::object $xstart; 
    bpy::object $xcount; bpy::object $xlimlo; bpy::object $xlimhi;
    bpy::object $gcount; bpy::object $glimlo; bpy::object $glimhi;
};

proxy_nlp_result 
minimize (
  bpy::object evalf, bpy::object gradf,
  bpy::object evalg, bpy::object gradg,
  bpy::object xstart,
  bpy::object xcount, bpy::object xlimlo, bpy::object xlimhi,
  bpy::object gcount, bpy::object glimlo, bpy::object glimhi,
  bpy::object options
) {

  coin::SmartPtr<proxy_nlp> problem = new proxy_nlp {
    evalf, gradf,
    evalg, gradg,
    xstart,
    xcount, xlimlo, xlimhi,
    gcount, glimlo, glimhi
  };

  // Process the options, swimmingly

  std::stringstream options_stream;
  auto options_iter_cur = bpy::stl_input_iterator<bpy::object>(options);
  auto options_iter_end = bpy::stl_input_iterator<bpy::object>();

  for (; options_iter_cur != options_iter_end; ++options_iter_cur) {
    std::string option_line = bpy::extract<std::string>(* options_iter_cur);

    options_stream << option_line;
    options_stream << std::endl;
  }

  //
  coin::ApplicationReturnStatus status;
  coin::SmartPtr<coin::IpoptApplication> app = IpoptApplicationFactory();

  // This is absolutely necessary to keep our sanity intact
  app->RethrowNonIpoptException(true);

  // Initialize the IPOPT application interface with the option stream,
  // permitting manual overrides below.
  status = app->Initialize(options_stream, true);
  
  // Update (force) some options: avoid printing the IPOPT banner
  app->Options()->SetStringValue("sb", "yes");

  // Update (force) some options: use BFGS for the Hessian matrix
  app->Options()->SetStringValue("hessian_approximation", "limited-memory");

  // Bang!
  status = app->OptimizeTNLP(problem);

  //
  return problem->$result;
}

/* Module entry point.
 * Registers types and functions with python runtime.
 */

BOOST_PYTHON_MODULE (ipopt4py) {

  // As per the documentation, 
  // the boost::python::numpy environment must be initialized.
  boost::python::numpy::initialize();

  boost::python::class_<proxy_nlp_result>("Result")
    .def_readonly("status",  & proxy_nlp_result::status)
    .def_readonly("message", & proxy_nlp_result::message)
    .def_readonly("success", & proxy_nlp_result::success)
    .def_readonly("fval",    & proxy_nlp_result::fval)
    .def_readonly("gval",    & proxy_nlp_result::gval)
    .def_readonly("xval",    & proxy_nlp_result::xval);

  boost::python::def("minimize", minimize);

  #ifdef IPOPT4PY_VERSION
  bpy::scope().attr("version") = std::string(IPOPT4PY_VERSION);
  #else
  bpy::scope().attr("version") = std::string(__TIMESTAMP__);
  #endif
}

