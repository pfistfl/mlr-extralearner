context("classif_kerasff")

test_that("classif_kerasff", {
  # requirePackagesOrSkip("RcppHNSW", default.method = "load")
  lrn = makeLearner("classif.kerasff")
  r = resample(lrn, iris.task, hout)
  expect_class(r, "ResampleResult")

  lrn = makeLearner("classif.kerasff", predict.type = "prob")
  r = resample(lrn, iris.task, hout)
  expect_class(r, "ResampleResult")


  lrn = makeLearner("classif.kerasff", optimizer = "sgd", momentum = 0.7, decay = 0.01, lr = 0.01)
  r = resample(lrn, iris.task, hout)
  expect_class(r, "ResampleResult")

  lrn = makeLearner("classif.kerasff", early_stopping_patience = 0)
  r = resample(lrn, iris.task, hout)

  lrn = makeLearner("classif.kerasff", optimizer = "adam", loss = "binary_crossentropy", nthread = 1L)
  r = resample(lrn, pid.task, hout)



  library(mlrMBO)
  library(randomsearch)
  ps = makeParamSet(
      makeIntegerParam(id = "epochs", lower = 10L, upper = 100L),
      makeDiscreteParam(id = "optimizer",
        values = c("sgd", "rmsprop", "adam", "nadam")),
      makeNumericParam(id = "lr", lower = -8, upper = 0, trafo = function(x) 5^x),
      makeNumericParam(id = "decay", lower = -8, upper = 0, trafo = function(x) 5^x),
      makeNumericParam(id = "momentum", lower = -8, upper = 0,trafo = function(x) 5^x,
        requires = quote(optimizer == "sgd")),
      makeNumericParam(id = "rho", lower = -8, upper = 0,trafo = function(x) 5^x,
        requires = quote(optimizer == "rmsprop")),
      makeNumericParam(id = "beta_1", lower = -8, upper = 0, trafo = function(x) 1 - 5^x,
        requires = quote(optimizer %in% c("adam", "nadam"))),
      makeNumericParam(id = "beta_2", lower = -8, upper = 0, trafo = function(x) 1 - 5^x,
        requires = quote(optimizer %in% c("adam", "nadam"))),
      makeIntegerParam(id = "layers", lower = 1L, upper = 4L),
      makeDiscreteParam(id = "batchnorm_dropout", values = c("batchnorm", "dropout", "none")),
      makeNumericParam(id = "input_dropout_rate", lower = 0, upper = 1, requires = quote(batchnorm_dropout == "dropout")),
      makeNumericParam(id = "dropout_rate", lower = 0, upper = 1, requires = quote(batchnorm_dropout == "dropout")),
      # Neurons / Layers
      makeIntegerParam(id = "units_layer1", lower = 1L, upper = 512),
      makeIntegerParam(id = "units_layer2", lower = 1L, upper = 512, requires = quote(layers >= 2)),
      makeIntegerParam(id = "units_layer3", lower = 1L, upper = 512, requires = quote(layers >= 3)),
      makeIntegerParam(id = "units_layer4", lower = 1L, upper = 512, requires = quote(layers >= 4)),
      # Activations
      makeDiscreteParam(id = "act_layer",
        values = c("elu", "relu", "selu", "tanh", "sigmoid")),
      # Initializers
      makeDiscreteParam(id = "init_layer",
        values = c("glorot_normal", "glorot_uniform", "he_normal", "he_uniform")),
      # Regularizers
      makeNumericParam(id = "l1_reg_layer",
        lower = -10, upper = -1, trafo = function(x) 5^x),
      makeNumericParam(id = "l2_reg_layer",
        lower = -10, upper = -1, trafo = function(x) 5^x),
      makeLogicalParam(id = "learning_rate_scheduler", default = FALSE)
    )
  tuneParams(lrn, sonar.task, hout, acc, makeTuneControlRandom(), par.set = ps)
})


