# This fixes an error in cubist, that occurs if the string "sample" is contained in the colnames.
# I guess this learner will become obsolete if this is fixed in Cubist itself.
#' @export
makeRLearner.regr.fixcubist = function() {
  makeRLearnerRegr(
    cl = "regr.fixcubist",
    package = "Cubist",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "committees", default = 1L, lower = 1L, upper = 100L),
      makeLogicalLearnerParam(id = "unbiased", default = FALSE),
      makeIntegerLearnerParam(id = "rules", default = 100L, lower = 1L),
      makeNumericLearnerParam(id = "extrapolation", default = 100, lower = 0, upper = 100),
      makeIntegerLearnerParam(id = "sample", default = 0L, lower = 0L),
      makeIntegerLearnerParam(id = "seed", default = sample.int(4096, size = 1) - 1L, tunable = FALSE),
      makeUntypedLearnerParam(id = "label", default = "outcome"),
      makeIntegerLearnerParam(id = "neighbors", default = 0L, lower = 0L, upper = 9L, when = "predict")
    ),
    properties = c("missings", "numerics", "factors"),
    name = "Cubist",
    short.name = "cubist",
    callees = c("cubist", "cubistControl", "predict.cubist")
  )
}

#' @export
trainLearner.regr.fixcubist = function(.learner, .task, .subset, .weights = NULL, unbiased, rules,
                                    extrapolation, sample, seed, label, ...) {
  ctrl = learnerArgsToControl(Cubist::cubistControl, unbiased, rules, extrapolation, sample, seed, label)

  d = getTaskData(.task, .subset, target.extra = TRUE)

  # Rename all sample variables to some random string if ocurring
  sample.vars = stringi::stri_subset_fixed(colnames(d$data), "sample", case_insensitive = TRUE)
  if (length(sample.vars) > 0) {
    new.names = stringi::stri_rand_strings(length(sample.vars), 10, '[a-zA-Z]')
    # Make sure we do not overwrite any existing varnames
    new.names = setdiff(new.names, colnames(d$data))[seq_len(length(sample.vars))]
    colnames(d$data)[colnames(d$data) %in% sample.vars] = new.names
  }
  m = Cubist::cubist(x = d$data, y = d$target, control = ctrl, ...)
  if (length(sample.vars) > 0) {
    m$new.names = new.names
    m$sample.vars = sample.vars
  }
  return(m)
}

#' @export
predictLearner.regr.fixcubist = function(.learner, .model, .newdata, ...) {
  # Overwrite colnames in newdata if sample variable names were replaced
  if (length(.model$learner.model$sample.vars) > 0)
    colnames(.newdata)[colnames(.newdata) %in% .model$learner.model$sample.vars] = .model$learner.model$new.names
  predict(.model$learner.model, newdata = .newdata, ...)
}
