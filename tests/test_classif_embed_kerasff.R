context("classif_kerasff")

test_that("classif_kerasff", {
  source("R/RLearner_classif_embed_kerasff_clean.R")
  library(OpenML)
  library(checkmate)
  library(mlrCPO)
  # Embed using keras embeddings:
  adult = convertOMLDataSetToMlr(getOMLDataSet(1590))
  lrn = cpoImputeMedian(affect.type = "numeric") %>>%
    cpoImputeConstant(affect.type = "factor", const = "_NA_") %>>%
    cpoScale(center = TRUE, scale = TRUE, affect.type = "numeric") %>>% 
    cpoEmbedclassif() %>>%
    makeLearner("classif.rpart")

  mod = train(lrn, adult); k_clear_session()

  # 1-Layer MLP
  makeLearner("classif.kerasff", layers = 1L, layer_1_units = 128)
})

context("regr_kerasff")

test_that("regr_kerasff", {
  source("R/RLearner_regr_embed_kerasff_clean.R")
  library(OpenML)
  library(checkmate)
  library(mlrCPO)
  # Embed using keras embeddings:
  adult = convertOMLDataSetToMlr(getOMLDataSet(1590))
  lrn = cpoImputeMedian(affect.type = "numeric") %>>%
    cpoImputeConstant(affect.type = "factor", const = "_NA_") %>>%
    cpoScale(center = TRUE, scale = TRUE, affect.type = "numeric") %>>% 
    cpoEmbedregr() %>>%
    makeLearner("regr.lm")

  mod = train(lrn, bh.task)

  # 1-Layer MLP
  mlp = cpoDummyEncode() %>>% makeLearner("regr.kerasff", layers = 1L, units_layer1 = 128)
  resample(mlp, bh.task, cv3, rmse)
})