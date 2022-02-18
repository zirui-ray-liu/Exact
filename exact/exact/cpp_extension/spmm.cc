#include <torch/torch.h>
#include <Python.h>
#include <torch/script.h>
#include "ext_common.h"
#include "spmm_cuda.h"


using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_fw(torch::Tensor rowptr, torch::Tensor col,
        torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
        std::string reduce) {
    TORCH_CHECK(rowptr.device().is_cuda());
    return spmm_cuda(rowptr, col, optional_value, mat, reduce);
}


torch::Tensor spmm_value_bw(torch::Tensor row, torch::Tensor rowptr,
                            torch::Tensor col, torch::Tensor mat,
                            torch::Tensor grad, std::string reduce) {
    TORCH_CHECK(rowptr.device().is_cuda());
    return spmm_value_bw_cuda(row, rowptr, col, mat, grad, reduce);
}

/****************************************/
/************** SPMM SUM ****************/
/****************************************/
torch::Tensor spmm_sum_fw(torch::optional<torch::Tensor> opt_row,
                          torch::Tensor rowptr, torch::Tensor col,
                          torch::optional<torch::Tensor> opt_value,
                          torch::optional<torch::Tensor> opt_colptr,
                          torch::optional<torch::Tensor> opt_csr2csc,
                          torch::Tensor mat) {
    bool has_value = opt_value.has_value();
    auto value = has_value ? opt_value.value() : col;
    if (has_value && torch::autograd::any_variable_requires_grad({value})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
    }
    if (torch::autograd::any_variable_requires_grad({mat})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
      AT_ASSERTM(opt_colptr.has_value(), "Argument `colptr` is missing");
      AT_ASSERTM(opt_csr2csc.has_value(), "Argument `csr2csc` is missing");
    }
    auto row = opt_row.has_value() ? opt_row.value() : col;
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : col;
    auto csr2csc = opt_csr2csc.has_value() ? opt_csr2csc.value() : col;
    torch::optional<torch::Tensor> opt_value_ = torch::nullopt;
    if (has_value)
      opt_value_ = value;
    auto out = std::get<0>(spmm_fw(rowptr, col, opt_value_, mat, "sum"));
    return out;
}

variable_list spmm_sum_bw(torch::Tensor row, torch::Tensor rowptr,
                          torch::Tensor col, Variable value, 
                          torch::Tensor colptr, torch::Tensor csr2csc, 
                          torch::Tensor mat, 
                          torch::Tensor grad_out,
                          bool has_value,
                          bool value_requires_grad,
                          bool mat_requires_grad){
    auto grad_value = Variable();
    if (has_value > 0 && value_requires_grad > 0) {
      grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "sum");
    }
    auto grad_mat = Variable();
    if (mat_requires_grad > 0) {
      torch::optional<torch::Tensor> opt_value = torch::nullopt;
      if (has_value)
        opt_value = value.index_select(0, csr2csc);

      grad_mat = std::get<0>(spmm_fw(colptr, row.index_select(0, csr2csc),
                                     opt_value, grad_out, "sum"));
    }
    return {grad_value, grad_mat};
}

/****************************************/
/************** SPMM MEAN ***************/
/****************************************/
torch::Tensor spmm_mean_fw(torch::optional<torch::Tensor> opt_row,
                           torch::Tensor rowptr, torch::Tensor col,
                           torch::optional<torch::Tensor> opt_value,
                           torch::optional<torch::Tensor> opt_rowcount,
                           torch::optional<torch::Tensor> opt_colptr,
                           torch::optional<torch::Tensor> opt_csr2csc,
                           torch::Tensor mat) {
    bool has_value = opt_value.has_value();
    auto value = has_value ? opt_value.value() : col;
    if (has_value && torch::autograd::any_variable_requires_grad({value})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
    }

    if (torch::autograd::any_variable_requires_grad({mat})) {
      AT_ASSERTM(opt_row.has_value(), "Argument `row` is missing");
      AT_ASSERTM(opt_rowcount.has_value(), "Argument `rowcount` is missing");
      AT_ASSERTM(opt_colptr.has_value(), "Argument `colptr` is missing");
      AT_ASSERTM(opt_csr2csc.has_value(), "Argument `csr2csc` is missing");
    }
    auto row = opt_row.has_value() ? opt_row.value() : col;
    auto rowcount = opt_rowcount.has_value() ? opt_rowcount.value() : col;
    auto colptr = opt_colptr.has_value() ? opt_colptr.value() : col;
    auto csr2csc = opt_csr2csc.has_value() ? opt_csr2csc.value() : col;

    torch::optional<torch::Tensor> opt_value_ = torch::nullopt;
    if (has_value)
      opt_value_ = value;
    auto out = std::get<0>(spmm_fw(rowptr, col, opt_value_, mat, "mean"));
    return out;
}

variable_list spmm_mean_bw(torch::Tensor row, torch::Tensor rowptr,
                           torch::Tensor col, Variable value, 
                           torch::Tensor rowcount, 
                           torch::Tensor colptr, torch::Tensor csr2csc,
                           torch::Tensor mat,
                           torch::Tensor grad_out, 
                           bool has_value,                 
                           bool value_requires_grad,
                           bool mat_requires_grad){
  auto grad_value = Variable();
  if (has_value > 0 && value_requires_grad > 0)
    grad_value = spmm_value_bw(row, rowptr, col, mat, grad_out, "mean");
  auto grad_mat = Variable();
  if (mat_requires_grad > 0) {
    row = row.index_select(0, csr2csc);
    rowcount = rowcount.toType(mat.scalar_type()).index_select(0, row);
    rowcount.clamp_(1);
    if (has_value > 0)
      rowcount = value.index_select(0, csr2csc).div(rowcount);
    else
      rowcount.pow_(-1);
    grad_mat = std::get<0>(spmm_fw(colptr, row, rowcount, grad_out, "sum"));
  }
  return {grad_value, grad_mat};
} 


/****************************************/
/************** SPMM MAX ****************/
/****************************************/
variable_list spmm_max_fw(Variable rowptr, Variable col, 
                          torch::optional<torch::Tensor> opt_value, 
                          Variable mat){
  bool has_value = opt_value.has_value();
  auto value = opt_value.has_value() ? opt_value.value() : col;
  torch::optional<torch::Tensor> opt_value_ = torch::nullopt;
  if (has_value)
    opt_value_ = value;
  auto result = spmm_fw(rowptr, col, opt_value_, mat, "max");
  auto out = std::get<0>(result);
  auto arg_out = std::get<1>(result).value();
  return {out, arg_out};
}

variable_list spmm_max_bw(Variable col, Variable value, 
                          Variable mat, Variable arg_out, 
                          torch::Tensor grad_out, 
                          bool has_value,                            
                          bool value_requires_grad,
                          bool mat_requires_grad){
  auto invalid_arg_mask = arg_out == col.size(0);
  arg_out = arg_out.masked_fill(invalid_arg_mask, 0);
  auto grad_value = Variable();
  if (has_value > 0 && value_requires_grad > 0) {
    auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
    auto out = mat.gather(-2, ind);
    out.mul_(grad_out);
    out.masked_fill_(invalid_arg_mask, 0);
    grad_value = torch::zeros_like(value);
    grad_value.scatter_add_(0, arg_out.flatten(), out.flatten());
  }
  auto grad_mat = Variable();
  if (mat_requires_grad > 0) {
    if (has_value > 0) {
      value = value.index_select(0, arg_out.flatten()).view_as(arg_out);
      value.mul_(grad_out);
    } else
      value = grad_out;

    value.masked_fill_(invalid_arg_mask, 0);
    auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
    grad_mat = torch::zeros_like(mat);
    grad_mat.scatter_add_(-2, ind, value);
  }
  return {grad_value, grad_mat};
}

/****************************************/
/************** SPMM MIN ****************/
/****************************************/
variable_list spmm_min_fw(Variable rowptr, Variable col, 
                          torch::optional<torch::Tensor> opt_value, 
                          Variable mat){

  bool has_value = opt_value.has_value();
  auto value = opt_value.has_value() ? opt_value.value() : col;
  torch::optional<torch::Tensor> opt_value_ = torch::nullopt;
  if (has_value)
    opt_value_ = value;
  auto result = spmm_fw(rowptr, col, opt_value_, mat, "min");
  auto out = std::get<0>(result);
  auto arg_out = std::get<1>(result).value();
  return {out, arg_out};
}

variable_list spmm_min_bw(Variable col, Variable value, 
                          Variable mat, Variable arg_out, 
                          torch::Tensor grad_out, 
                          bool has_value,
                          bool value_requires_grad,
                          bool mat_requires_grad){
  auto invalid_arg_mask = arg_out == col.size(0);
  arg_out = arg_out.masked_fill(invalid_arg_mask, 0);
  auto grad_value = Variable();
  if (has_value > 0 && value_requires_grad > 0) {
    auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
    auto out = mat.gather(-2, ind);
    out.mul_(grad_out);
    out.masked_fill_(invalid_arg_mask, 0);
    grad_value = torch::zeros_like(value);
    grad_value.scatter_add_(0, arg_out.flatten(), out.flatten());
  }
  auto grad_mat = Variable();
  if (mat_requires_grad > 0) {
    if (has_value > 0) {
      value = value.index_select(0, arg_out.flatten()).view_as(arg_out);
      value.mul_(grad_out);
    } else
      value = grad_out;

    value.masked_fill_(invalid_arg_mask, 0);
    auto ind = col.index_select(0, arg_out.flatten()).view_as(arg_out);
    grad_mat = torch::zeros_like(mat);
    grad_mat.scatter_add_(-2, ind, value);
  }
  return {grad_value, grad_mat};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_sum_fw", &spmm_sum_fw);
  m.def("spmm_sum_bw", &spmm_sum_bw);
  m.def("spmm_mean_fw", &spmm_mean_fw);
  m.def("spmm_mean_bw", &spmm_mean_bw);
  m.def("spmm_max_fw", &spmm_max_fw);
  m.def("spmm_max_bw", &spmm_max_bw);
  m.def("spmm_min_fw", &spmm_min_fw);
  m.def("spmm_min_bw", &spmm_min_bw);
}