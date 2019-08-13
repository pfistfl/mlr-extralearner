#' @export
makeRLearner.classif.embed_kerasff = function() {
  makeRLearnerClassif(
    cl = "classif.embed_kerasff",
    package = "keras",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "epochs", lower = 1L, default = 30L),
      makeIntegerLearnerParam(id = "early_stopping_patience", lower = 0L, default = 1L),
      makeDiscreteLearnerParam(id = "optimizer",  default = "sgd",
        values = c("sgd", "rmsprop", "adagrad", "adadelta", "adam", "nadam")),
      makeNumericLearnerParam(id = "lr", lower = 0, upper = 1, default = 0.001),
      makeNumericLearnerParam(id = "decay", lower = 0, upper = 1, default = 0),
      makeNumericLearnerParam(id = "momentum", lower = 0, upper = 1, default = 0,
        requires = quote(optimizer == "sgd")),
      makeNumericLearnerParam(id = "rho", lower = 0, upper = 1, default = 0.001,
        requires = quote(optimizer == "rmsprop")),
      makeNumericLearnerParam(id = "beta_1", lower = 0, upper = 1, default = 0.9,
        requires = quote(optimizer %in% c("adam", "nadam"))),
      makeNumericLearnerParam(id = "beta_2", lower = 0, upper = 1, default = 0.999,
        requires = quote(optimizer %in% c("adam", "nadam"))),
      makeDiscreteLearnerParam(id = "loss",
        values = c("categorical_crossentropy", "sparse_categorical_crossentropy"),
        default = "categorical_crossentropy"),
      makeIntegerLearnerParam(id = "batch_size", lower = 1L, upper = Inf, default = 1L),
      makeIntegerLearnerParam(id = "n_layers", lower = 1L, upper = 4L, default = 1L),
      makeDiscreteLearnerParam(id = "batchnorm_dropout",
        values = c("batchnorm", "dropout", "none"), default = "none"),
      makeNumericLearnerParam(id = "input_dropout_rate", default = 0, lower = 0, upper = 1, requires = quote(batchnorm_dropout == "dropout")),
      makeNumericLearnerParam(id = "dropout_rate", default = 0, lower = 0, upper = 1, requires = quote(batchnorm_dropout == "dropout")),
      # Neurons / Layers
      makeIntegerLearnerParam(id = "units_layer1", lower = 1L, default = 1L),
      makeIntegerLearnerParam(id = "units_layer2", lower = 1L, default = 1L,
        requires = quote(layers >= 2)),
      makeIntegerLearnerParam(id = "units_layer3", lower = 1L, default = 1L,
        requires = quote(layers >= 3)),
      makeIntegerLearnerParam(id = "units_layer4", lower = 1L, default = 1L,
        requires = quote(layers >= 4)),
      # Activations
      makeDiscreteLearnerParam(id = "act_layer",
        values = c("elu", "relu", "selu", "tanh", "sigmoid","PRelU", "LeakyReLu"),
        default = "relu"),
      # Initializers
      makeDiscreteLearnerParam(id = "init_layer",
        values = c("glorot_normal", "glorot_uniform", "he_normal", "he_uniform"),
        default = "glorot_uniform"),
      # Regularizers
      makeNumericLearnerParam(id = "l1_reg_layer",
        lower = 0, upper = 1, default = 0),
      makeNumericLearnerParam(id = "l2_reg_layer",
        lower = 0, upper = 1, default = 0),
      makeNumericLearnerParam(id = "validation_split",
        lower = 0, upper = 1, default = 0),
      makeLogicalLearnerParam(id = "learning_rate_scheduler", default = FALSE)
    ),
    properties = c("numerics", "factors", "prob", "twoclass", "multiclass"),
    par.vals = list(),
    name = "Keras Fully-Connected NN",
    short.name = "kerasff"
  )
}


trainLearner.classif.embed_kerasff  = function(.learner, .task, .subset, .weights = NULL,
  epochs = 30L, early_stopping_patience = 10L, learning_rate_scheduler = FALSE,
  optimizer = "adam", lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, momentum = 0, decay = 0.1,
  rho = 0.9, batch_size = 128L, embed_dropout_rate = 0.2, dropout_rate = 0.5,
  units_layer1 = 32, units_layer2 = 32, units_layer3 = 32, units_layer4 = 32,
  n_layers = 2L,
  l1_reg_layer = 0, l2_reg_layer = 0, validation_split = 0.2) {

  require("keras")
  keras = reticulate::import("keras")
  input_shape = getTaskNFeats(.task)
  output_shape = length(getTaskClassLevels(.task))
  data = getTaskData(.task, .subset, target.extra = TRUE)
  target_levels = levels(data$target)

  regularizer = regularizer_l1_l2(l1 = l1_reg_layer, l2 = l2_reg_layer)
  optimizer = switch(optimizer,
    "sgd" = optimizer_sgd(lr, momentum, decay = decay),
    "rmsprop" = optimizer_rmsprop(lr, rho, decay = decay),
    "adagrad" = optimizer_adagrad(lr, decay = decay),
    "adam" = optimizer_adam(lr, beta_1, beta_2, decay = decay),
    "nadam" = optimizer_nadam(lr, beta_1, beta_2, schedule_decay = decay)
  )

  callbacks = c()
  # callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
  if (early_stopping_patience > 0)
    callbacks = c(callbacks, callback_early_stopping(monitor = 'val_loss', patience = early_stopping_patience))
  if (learning_rate_scheduler) {
    clr = function(x) ((sin(x/300)+1)*exp(-x/10000)/100)
    callback_lr_init <- function(x){
          iter <<- 0
          lr_hist <<- c()
    }
    callback_lr_set <- function(batch, logs){
          iter <<- iter + 1
          learning_r <- clr(iter)
          k_set_value(model$optimizer$lr, learning_r)
    }
    callback_lr_log <- function(batch, logs){
          # k_get_value(): https://keras.rstudio.com/articles/backend.html#backend-functions
          lr_hist <<- c(lr_hist, k_get_value(model$optimizer$lr))
          #cat(paste0("iter: ",iter," - lr: ",k_get_value(model$optimizer$lr),"\n"))
    }
    callback_lr = callback_lambda(on_train_begin=callback_lr_init, on_batch_begin=callback_lr_set)
    callback_logger = callback_lambda(on_batch_begin=callback_lr_log)
    callbacks = c(callbacks, callback_lr, callback_logger)
  }

  # --- Build Up Model -------------------------------------------------------
  units_layers = c(units_layer1, units_layer2, units_layer3, units_layer4)
  if (output_shape == 2) output_shape = 2

  # The model consists of Embedding layers for the categorical variables,
  # followed by a Dropout of emb_drop, and a BatchNorm for the continuous variables.
  # The results are concatenated and followed by blocks of BatchNorm, Dropout, 
  # Linear and ReLU (the first block skips BatchNorm and Dropout, the last block skips
  # the ReLU).
  embedding = make_embedding(data$data, embed_dropout = embed_dropout_rate)
  layers = embedding$layers

  for (i in seq_len(n_layers + 1L)) {
    if (i > 1) {
      layers = layers %>%
        layer_batch_normalization() %>%
        layer_dropout(dropout_rate)
    }
    # Final layer 
    if (i < n_layers + 1)
    layers = layers %>%
      layer_dense(units = units_layers[i], kernel_regularizer = regularizer) %>%
      layer_activation_relu()
    else 
      layers = layers %>% layer_dense(units = output_shape) %>% layer_activation_softmax()
  }

  model = keras_model(input = embedding$inputs, output = layers)

  # --- Compile and Fit ---------------------------------------------------------------
  data = reshape_data_embedding(data$data, data$target)

  if (output_shape == 2) {
    model %>% compile(
      optimizer = "adam",
      loss =  "binary_crossentropy",
      metrics = "accuracy"
    )
    history = model %>% fit(
      x = data$data,
      y = to_categorical(data$label),
      # steps_per_epoch = nrow(data$data) %/% batch_size,
      batch_size = batch_size,
      epochs  = epochs,
      validation_split = validation_split,
      callback = callbacks
    )
  } else  {
    model %>% compile(
      optimizer = optimizer,
      loss =  "categorical_crossentropy",
      metrics = "accuracy"
    )
    history = model %>% fit(
      x = data$data,
      y = to_categorical(data$label, num_classes = output_shape),
      # steps_per_epoch = nrow(data$data) %/% batch_size,
      batch_size = batch_size,
      epochs  = epochs,
      validation_split = validation_split
    )
  }

  return(list(model = model, history = history, target_levels = target_levels, data = data, history = history))
}

predictLearner.classif.embed_kerasff = function(.learner, .model, .newdata, ...) {
  newdata = reshape_data_embedding(.newdata, target = NULL)$data 
  p = .model$learner.model$model %>% predict(newdata)
  if (.learner$predict.type == "prob") {
    colnames(p) = .model$learner.model$target_levels
  } else {
    argmax = apply(p, 1, which.max)
    p = as.factor(.model$learner.model$target_levels[argmax])
  }
  return(p)
}

reshape_data_embedding = function(data, target) {
  assert_factor(target, null.ok = TRUE)
  assert_data_frame(data)
  type = BBmisc::vcapply(data, function(x) class(x)[[1]])
  embed_vars = type %in% c("ordered", "factor")

  out_data = list()
  if (sum(embed_vars)  > 0)
    out_data = setNames(lapply(as.list(data[, embed_vars]), function(x) as.integer(x) - 1L), colnames(data)[embed_vars])
  if (sum(!embed_vars) > 0)
    out_data$continuous = as.matrix(data[, !embed_vars])
  
  if (is.null(target)) list(data = out_data)
  else list(data = out_data,
    label =  array(as.integer(target) - 1, dim = c(nrow(data), 1)))
}

make_embedding = function(data, embed_size = NULL, embed_dropout = 0) {
  assert_data_frame(data)
  assert_numeric(embed_size, null.ok = TRUE)
  assert_number(embed_dropout)
  type = BBmisc::vcapply(data, function(x) class(x)[[1]])
  embed_vars = type %in% c("ordered", "factor")
  n_cont = length(type[!embed_vars])

  # Embeddings for categorical variables
  embds = list()
  if (sum(embed_vars) > 0) {
    embds = Map(function(x, feat_name) {
      n_cat = length(levels(x))
      # Use heuristic from fast.ai https://github.com/fastai/fastai/blob/master/fastai/tabular/data.py
      if (length(embed_size) >= 2) embed_size = embed_size[feat_name]
      if (length(embed_size) == 0) embed_size = min(600L, round(1.6 * n_cat^0.56))
      input = layer_input(shape = 1, dtype = "int32", name = feat_name) 
      layers = input %>%
      layer_embedding(input_dim = as.numeric(n_cat), output_dim = as.numeric(embed_size), input_length = 1, name = paste0("embed_", feat_name)) %>%
      layer_dropout(embed_dropout, input_shape = as.numeric(embed_size)) %>%
      layer_flatten()
      return(list(input = input, layers = layers))
    }, data[, embed_vars, drop = FALSE], names(type[embed_vars]))
  }
  # Layer for the continuous variables
  if (n_cont > 0) {
    input = layer_input(shape = n_cont, dtype = "float32", name = "continuous") 
    layers = input %>% layer_batch_normalization(input_shape = n_cont, axis = -1)
    embds = c(embds, list(cont = list(input = input, layers = layers)))
  }

  # Concatenate in case
  if (length(embds) >= 2) 
    layers = layer_concatenate(unname(lapply(embds, function(x) x$layers)))
  else
    layers = unname(embds[[1]]$layers)
   return(list(inputs = lapply(embds, function(x) x$input), layers = layers))
}

get_embeddings = function(model, data) {
  assert_data_frame(data)
  assert_class(model, "keras.engine.training.Model")
  
  type = BBmisc::vcapply(data, function(x) class(x)[[1]])
  embed_vars = type %in% c("ordered", "factor")
  fct_lvls = lapply(data[, embed_vars], function(x) {levels(x)})
  names(fct_lvls) = paste0("embed_", names(type[embed_vars]))

  layers = sapply(model$layers, function(x) x$name)
  embed_layers = layers[grepl("embed", layers)]

  wts = setNames(lapply(embed_layers, 
    function(layer) {
      wt = get_layer(model, layer)$get_weights()[[1]]
      colnames(wt) = paste0("embed_", layer, seq_len(ncol(wt)))
      rownames(wt) = fct_lvls[[layer]]
      return(wt)
    }), embed_layers)
  return(wts)
}


# http://thecooldata.com/2019/01/learning-rate-finder-with-cifar10-keras-r/
find_lr = function(mlr_mod, batch_size = 128, n_epochs = 5) {

  res = mlr_mod$learner.model$next.model$learner.model
  model = res$model
  n = round(nrow(res$data$label) * 0.9)

  LogMetrics <- R6::R6Class("LogMetrics",
    inherit = KerasCallback,
    public = list(
      loss = NULL,
      acc = NULL,
      on_batch_end = function(batch, logs=list()) {
        self$loss <- c(self$loss, logs[["loss"]])
        self$acc <- c(self$acc, logs[["acc"]])
      }
  ))
  callback_lr_init <- function(logs){
    iter <<- 0
    lr_hist <<- c()
    iter_hist <<- c()
  }
  callback_lr_set <- function(batch, logs){
    lr_max <- 0.1
    n_iter <- ceiling(n_epochs * (n / batch_size))
    growth_constant <- 15
    lr_max = 0.1
    l_rate <- exp(seq(0, growth_constant, length.out=n_iter))
    l_rate <- l_rate/max(l_rate)
    l_rate <- l_rate * lr_max
    iter <<- iter + 1
    LR <- l_rate[iter] # if number of iterations > l_rate values, make LR constant to last value
    if(is.na(LR)) LR <- l_rate[length(l_rate)]
    k_set_value(model$optimizer$lr, LR)
  }
  callback_lr_log <- function(batch, logs){
    lr_hist <<- c(lr_hist, k_get_value(model$optimizer$lr))
    iter_hist <<- c(iter_hist, k_get_value(model$optimizer$iterations))
  }
  callback_lr <- callback_lambda(on_train_begin=callback_lr_init, on_batch_begin=callback_lr_set)
  callback_logger <- callback_lambda(on_batch_end=callback_lr_log)
  callback_log_acc_lr <- LogMetrics$new()

  history <- model %>% fit(
      x = res$data$data,
      y =  to_categorical(res$data$label),
      batch_size = batch_size,
      epochs = n_epochs,
      shuffle = TRUE,
      callbacks = list(callback_lr, callback_logger, callback_log_acc_lr),
      verbose = 2)
  # plot(lr_hist, callback_log_acc_lr$acc, log="x", type="b", pch=16, cex=0.3, xlab="learning rate (log scale)", ylab="accuracy")
  plot(zoo::rollmean(lr_hist, 100), zoo::rollmean(callback_log_acc_lr$acc, 100), log="x", type="l", pch=16, cex=0.3, xlab="learning rate", ylab="accuracy: rollmean(100)")
  return(history)
}


if (FALSE) {
  library(OpenML)
  library(checkmate)
  library(mlrCPO)
  library(ggplot2)
  library(dplyr)
  adult = convertOMLDataSetToMlr(getOMLDataSet(1590))

  lrn = cpoImputeMedian(affect.type = "numeric") %>>%
    cpoImputeMode(affect.type = "factor") %>>%
    cpoScale(center = TRUE, scale = TRUE, affect.type = "numeric") %>>%
    makeLearner("classif.embed_kerasff",
      lr = 10^-5, epochs = 5, units_layer1 = 128, units_layer2 = 64, units_layer3 = 64, n_layers = 3L,
      validation_split = 0.1, early_stopping_patience = 5)
  mod = train(lrn, adult)

  #Get and plot embeddings
  res = mod$learner.model$next.model$learner.model
  model = res$model
  embds = get_embeddings(model, getTaskData(adult))

  data.frame(prcomp(embds$embed_race)$x[, c(1,2)]) %>%
  mutate(name = rownames(.)) %>%
  ggplot() + geom_text(aes(x = PC1, y = PC2, label = name))

  # Learning Rate finder
  find_lr(mod)

  # --- 2071 Adult
  tsk = getOMLTask(2071)
  lrn = makeLearner("classif.embed_kerasff",
      lr = 10^-5, epochs = 25, units_layer1 = 64, units_layer2 = 64, units_layer3 = 64, n_layers = 3L,
      validation_split = 0.1, early_stopping_patience = 5)
  lrn = makePreprocWrapperCaret(
    makeImputeWrapper(lrn,
      classes = list(integer = imputeMedian(), numeric = imputeMedian(), factor = imputeConstant("_NA_")),
      dummy.classes = c("numeric", "integer") 
    ), "ppc.scale" = TRUE, "ppc.center" = TRUE)
  runTaskMlr(tsk, lrn)

  # --- 14966 Ozone
  tsk = getOMLTask(145855)
  lrn = makeLearner("classif.embed_kerasff",
      lr = 10^-3, epochs = 50, units_layer1 = 1024, units_layer2 = 512, units_layer3 = 256, n_layers = 3L,
      validation_split = 0.1, early_stopping_patience = 10)
  lrn = makePreprocWrapperCaret(
    makeImputeWrapper(lrn,
      classes = list(integer = imputeMedian(), numeric = imputeMedian(), factor = imputeConstant("_NA_")),
      dummy.classes = c("numeric", "integer") 
    ), "ppc.scale" = TRUE, "ppc.center" = TRUE)
  runTaskMlr(tsk, lrn)

}
