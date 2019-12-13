#' @export
makeRLearner.classif.embed_kerasff = function() {
  makeRLearnerClassif(
    cl = "classif.embed_kerasff",
    package = "keras",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "epochs", lower = 0L, default = 30L),
      makeIntegerLearnerParam(id = "early_stopping_patience", lower = 0L, default = 1L),
      makeDiscreteLearnerParam(id = "optimizer",  default = "adam",
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
      makeIntegerLearnerParam(id = "batch_size", lower = 1L, upper = Inf, default = 1L),
      makeIntegerLearnerParam(id = "n_layers", lower = 1L, upper = 4L, default = 1L),
      makeNumericLearnerParam(id = "dropout_rate", default = 0, lower = 0, upper = 1),
      makeNumericLearnerParam(id = "embed_dropout_rate", default = 0.05, lower = 0, upper = 1),
      # Neurons / Layers
      makeIntegerLearnerParam(id = "units_layer1", lower = 1L, default = 1L),
      makeIntegerLearnerParam(id = "units_layer2", lower = 1L, default = 1L,
        requires = quote(layers >= 2)),
      makeIntegerLearnerParam(id = "units_layer3", lower = 1L, default = 1L,
        requires = quote(layers >= 3)),
      makeIntegerLearnerParam(id = "units_layer4", lower = 1L, default = 1L,
        requires = quote(layers >= 4)),
      makeNumericLearnerParam(id = "validation_split", lower = 0, upper = 1, default = 0),
      makeLogicalLearnerParam(id = "learning_rate_scheduler", default = TRUE),
      makeUntypedLearnerParam(id = "callbacks", default = c()),
      makeLogicalLearnerParam(id = "mixup", default = FALSE),
      makeLogicalLearnerParam(id = "smooth_labels", default = FALSE)
    ),
    properties = c("numerics", "factors", "prob", "twoclass", "multiclass"),
    par.vals = list(),
    name = "Keras Fully-Connected NN",
    short.name = "kerasff"
  )
}


trainLearner.classif.embed_kerasff  = function(.learner, .task, .subset, .weights = NULL,
  epochs = 10L, early_stopping_patience = 5L, batch_size = 256L,
  learning_rate_scheduler = TRUE, validation_split = 0.2,
  optimizer = "adam", lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01,
  momentum = 0, rho = 0.9,
  embed_dropout_rate = 0.05, dropout_rate = 0.4,
  n_layers = 3L,
  units_layer1 = 512, units_layer2 = 256, units_layer3 = 128, units_layer4 = 64,
  tensorboard = FALSE,
  callbacks = c(), mixup = FALSE, smooth_labels = FALSE, mixup_factor = 2) {

  require("keras")
  keras = reticulate::import("keras")
  assert_flag(mixup)
  assert_flag(smooth_labels)
  assert_count(early_stopping_patience)
  assert_flag(learning_rate_scheduler)

  input_shape = getTaskNFeats(.task)
  output_shape = length(getTaskClassLevels(.task))
  data = getTaskData(.task, .subset, target.extra = TRUE)
  target_levels = levels(data$target)
  optimizer = switch(optimizer,
    "sgd" = optimizer_sgd(lr, momentum, decay = decay),
    "rmsprop" = optimizer_rmsprop(lr, rho, decay = decay),
    "adagrad" = optimizer_adagrad(lr, decay = decay),
    "adam" = optimizer_adam(lr, beta_1, beta_2, decay = decay, clipnorm = 1),
    "nadam" = optimizer_nadam(lr, beta_1, beta_2, schedule_decay = decay)
  )

  if (early_stopping_patience > 0)
    callbacks = c(callbacks, callback_early_stopping(monitor = 'val_loss', patience = early_stopping_patience))

  if (tensorboard) callbacks = c(callbacks, callback_tensorboard())

  if (learning_rate_scheduler) {
    n_batches = (1 - validation_split) * ceiling(getTaskSize(.task) / batch_size)
    # Trying out different learning rate schedulers
    # clr = function(x) ((sin(x / n_batches * 2 * pi) + 1)*exp(-x/5000) + .01)
    # clr = function(x) (min(10^-3, (sin(x / (epochs*n_batches) * pi))*exp(-x/5000) + .01))
    make_saw = function(x, max_x, lin = 0.3) {
      if (x <= lin * max_x) x / (lin * max_x)
      else (cos((x - lin * max_x) / max_x * pi / (1 - lin)) + 1) / 2
    }
    clr = function(x, n_saws = 1) {
      max_x = epochs*n_batches
      lrs = make_saw(x %% (max_x / n_saws), (max_x / n_saws))
      lrs + 10^-8
    }
    callback_lr_init = function(x){
      iter <<- 0
      lr_hist <<- c()
    }
    callback_lr_set = function(batch, logs){
      iter <<- iter + 1
      learning_r = clr(iter)
      k_set_value(model$optimizer$lr, lr * learning_r)
    }
    callback_lr_log = function(batch, logs){
      lr_hist <<- c(lr_hist, k_get_value(model$optimizer$lr))
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
  # Keras requires "input" (= input layers) and "output" (additional layers)
  # for model construction.
  embedding = make_embedding(data$data, embed_dropout = embed_dropout_rate)
  layers = embedding$layers
  for (i in seq_len(n_layers + 1L)) {
    if (i > 1) {
      layers = layers %>%
        layer_batch_normalization(epsilon = 0.01) %>%
        layer_dropout(dropout_rate)
    }
    # Final layer
    if (i < n_layers + 1)
    layers = layers %>%
      layer_dense(units = units_layers[i], kernel_initializer = initializer_he_uniform()) %>%
      layer_activation_leaky_relu(alpha = 0.3)
    else
      layers = layers %>% layer_dense(units = output_shape) %>% layer_activation_softmax()
  }
  model = keras_model(input = embedding$inputs, output = layers)

  # --- Compile and Fit ---------------------------------------------------------------
  # data has to be a list with 1 element per input. continuous vars are in a
  # list element "continuous".
  data = reshape_data_embedding(data$data, data$target)

  # Add mixup or label smoothing?
  if (mixup) data = mixup_data(data, factor = mixup_factor)
  else data$label = to_categorical(data$label)
  if (smooth_labels) data$label = smooth_labels(data$label, alpha = 0.95)

  if (output_shape == 2) loss = "binary_crossentropy"
  else loss = "categorical_crossentropy"

  model %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = "categorical_accuracy"
  )

  if (epochs > 0) {
    history = model %>% fit(
      x = data$data,
      y = data$label,
      batch_size = batch_size,
      epochs  = epochs,
      validation_split = validation_split,
      callbacks = callbacks
    )
  } else history = NULL

  return(list(model = model, history = history, target_levels = target_levels, data = data,
    history = history, fct_levels = data$fct_levels))
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
  k_clear_session()
  return(p)
}


#' Reshape a dataset for use with entity embeddings.
#' continuous vars are stored in a matrix "continuous"
#' every categorical variable is integer encoded and stored
#' as a single list element.
reshape_data_embedding = function(data, target) {
  assert_factor(target, null.ok = TRUE)
  assert_data_frame(data)
  type = BBmisc::vcapply(data, function(x) class(x)[[1]])
  embed_vars = type %in% c("ordered", "factor")

  fct_levels = lapply(as.list(data[, embed_vars]), function(x) levels(x))
  out_data = list()
  if (sum(embed_vars)  > 0)
    out_data = setNames(lapply(as.list(data[, embed_vars]), function(x) as.integer(x) - 1L), colnames(data)[embed_vars])
  if (sum(!embed_vars) > 0)
    out_data$continuous = as.matrix(data[, !embed_vars])

  if (is.null(target)) list(data = out_data, fct_levels = fct_levels)
  else list(
    data = out_data, fct_levels = fct_levels,
    label =  array(as.integer(target) - 1, dim = c(nrow(data), 1))
    )
}

# Create the embedding for a dataset.
# Creates an input for each categorical var, concatenates those,
# Adds batch-norm to continuous vars etc.
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
      layer_embedding(input_dim = as.numeric(n_cat), output_dim = as.numeric(embed_size),
        input_length = 1, name = paste0("embed_", feat_name),
        embeddings_initializer = initializer_he_uniform()) %>%
      layer_dropout(embed_dropout, input_shape = as.numeric(embed_size)) %>%
      layer_flatten()
      return(list(input = input, layers = layers))
    }, data[, embed_vars, drop = FALSE], names(type[embed_vars]))
  }
  # Layer for the continuous variables
  if (n_cont > 0) {
    input = layer_input(shape = n_cont, dtype = "float32", name = "continuous")
    layers = input %>% layer_batch_normalization(input_shape = n_cont, axis = 1)
    embds = c(embds, list(cont = list(input = input, layers = layers)))
  }

  # Concatenate in case
  if (length(embds) >= 2)
    layers = layer_concatenate(unname(lapply(embds, function(x) x$layers)))
  else
    layers = unname(embds[[1]]$layers)
   return(list(inputs = lapply(embds, function(x) x$input), layers = layers))
}

smooth_labels = function(labels, alpha = 0.9) {
  if (alpha == 1) return(labels)
  t(apply(labels, 1, function(x) {
    alpha * x + (1 - alpha) / length(x)
  }))
}

mixup_data = function(data, factor, alpha) {
  assert_true(factor >= 1)
  if (factor == 1) return(data)
  n = nrow(data$label)
  n_points = ceiling(n * factor - n)
  alpha = rbeta(n_points, shape1 = 0.1, shape2 = 0.1)
  ids = sample(seq_len(n), n_points*2, replace = TRUE)
  labs = to_categorical(data$label[ids,])
  data$data[names(data$data) != "continuous"] = data.frame(lapply(data$data[names(data$data) != "continuous"],
    function(x) {
      pairs = split(ids, seq_len(ceiling(n_points)))
      z = unlist(Map(
        function(y, alpha) {sample(y, 1, prob = c(alpha, 1-alpha))},
        pairs, alpha
      ))
      c(x, x[z])
  }))
  data$data$continuous = rbind(
    data$data$continuous,
    t(sapply(seq_len(n_points), function(i) {
      idx = ids[c(i*2 - 1, i*2)]
      cont = t(as.matrix(data$data$continuous[idx, ])) %*% matrix(c(alpha[i], 1-alpha[i]), nrow = 2)
    }))
  )
  data$label = rbind(
    to_categorical(data$label),
    t(sapply(seq_len(n_points), function(i) {
      t(as.matrix(labs[c(i*2 - 1, i*2), ])) %*% matrix(c(alpha[i], 1-alpha[i]), nrow = 2)
    }))
  )
  return(data)
}


#' Return data with embeddings.
#' @param model mlr mdoel
#' @param data data.frame, getTaskData(task)
#' @param na_level Level for missing values chosen in the embedding.
#' @return data.frame data with levels instead of features
embed_with_model = function(model, data, na_level = "_NA_") {
  assert_data_frame(data)
  assert_class(model, "WrappedModel")
  assert_string(na_level)

  wts = get_embeddings(model)
  lst = lapply(names(wts), function(x) {
    fct = data[, gsub("embed_", "", x)]
    fct = addNA(fct)
    levels(fct)[is.na(levels(fct))] = na_level
    wts[[x]][fct, ]
  })
  data = data[, - which(colnames(data) %in% gsub("embed_", "", names(wts)))]
  data = cbind(data, do.call("cbind", lst))
  return(data)
}

get_embeddings = function(model) {
  assert_class(model, "WrappedModel")
  learner_model = mlr:::getLearnerModel(model, more.unwrap = TRUE)
  model = learner_model$model
  fct_levels = learner_model$fct_levels
  names(fct_levels) = paste0("embed_", names(fct_levels))

  layers = sapply(model$layers, function(x) x$name)
  embed_layers = layers[grepl("embed", layers)]

  wts = setNames(lapply(embed_layers,
    function(layer) {
      wt = get_layer(model, layer)$get_weights()[[1]]
      colnames(wt) = paste0(layer, seq_len(ncol(wt)))
      rownames(wt) = fct_levels[[layer]]
      return(wt)
    }), embed_layers)
  return(wts)
}


# Implementation of a learning rate finder.
# This is a wrapper around find_lr, that creates the model
# and returns the learning rate.
lr_finder = function(lrn, tsk, epochs = 5) {
  lrn = setHyperPars(lrn, epochs = 0)
  mod = train(lrn, tsk)
  find_lr(mod, epochs = epochs)
}

# Adapted from
# http://thecooldata.com/2019/01/learning-rate-finder-with-cifar10-keras-r/
find_lr = function(mlr_mod, batch_size = 128, epochs = 5) {
  requireNamespace("zoo")
  res = getLearnerModel(mlr_mod, more.unwrap = TRUE)
  model = res$model
  n = nrow(res$data$label)

  LogMetrics = R6::R6Class("LogMetrics",
    inherit = KerasCallback,
    public = list(
      loss = NULL,
      acc = NULL,
      on_batch_end = function(batch, logs=list()) {
        self$loss = c(self$loss, logs[["loss"]])
        self$acc = c(self$acc, logs[["acc"]])
      }
  ))

  # Define calbacks
  callback_lr_init = function(logs){
    iter <<- 0
    lr_hist <<- c()
    iter_hist <<- c()
  }
  callback_lr_set = function(batch, logs){
    lr_max = .8
    n_iter = epochs * ceiling((n / batch_size))
    l_rate = exp(seq(0, 15, length.out = n_iter))
    l_rate = l_rate / max(l_rate)
    l_rate = l_rate * lr_max
    iter <<- iter + 1
    LR = l_rate[iter] # if number of iterations > l_rate values, make LR constant to last value
    if(is.na(LR)) LR = l_rate[length(l_rate)]
    k_set_value(model$optimizer$lr, LR)
  }
  callback_lr_log = function(batch, logs){
    lr_hist <<- c(lr_hist, k_get_value(model$optimizer$lr))
    iter_hist <<- c(iter_hist, k_get_value(model$optimizer$iterations))
  }
  callback_lr = callback_lambda(on_train_begin=callback_lr_init, on_batch_begin=callback_lr_set)
  callback_logger = callback_lambda(on_batch_end=callback_lr_log)
  callback_log_acc_lr = LogMetrics$new()

  history = model %>% fit(
      x = res$data$data,
      y =  res$data$label,
      batch_size = batch_size,
      epochs = epochs,
      shuffle = TRUE,
      callbacks = list(callback_lr, callback_logger, callback_log_acc_lr),
      verbose = 2)
  k_clear_session()
  plot(zoo::rollmean(lr_hist, 100), zoo::rollmean(callback_log_acc_lr$loss, 100), log="x", type="l", pch=16, cex=0.3, xlab="learning rate", ylab="loss: rollmean(100)")
  return(history)
}









if (FALSE) {
  # Some tests, checks and example code
  library(OpenML)
  library(checkmate)
  library(mlrCPO)
  library(keras)
  adult = convertOMLDataSetToMlr(getOMLDataSet(1590))
  lrn = cpoImputeMedian(affect.type = "numeric") %>>%
    cpoImputeConstant(affect.type = "factor", const = "_NA_") %>>%
    cpoScale(center = TRUE, scale = TRUE, affect.type = "numeric") %>>%
    makeLearner("classif.embed_kerasff", lr = 0.1, epochs = 5,
      units_layer3 = 128, units_layer2 = 128, units_layer1 = 128,
      decay = 0.1)
  k_clear_session()
  mod = train(lrn, adult)

  # Plot history
  res = mlr:::getLearnerModel(mod, more.unwrap = TRUE)
  plot(res$history)

  # Plot weight stats (needs callback "callback_wt_stats")
  library(ggplot2)
  ggplot(do.call("rbind", wt_stats)) +
    geom_point(aes(x = mean, y = sd)) +
    facet_grid(iter~layer, scales = "free")

  # Get the embeddings
  model = res$model
  embds = get_embeddings(model, getTaskData(adult))

  resample("classif.ranger", adult, hout)
  resample("classif.ranger", makeClassifTask(data = data_embd, target = "class"), hout)

  resample(
    makePreprocWrapperCaret(
    makeImputeWrapper("classif.ranger",
      classes = list(integer = imputeMedian(), numeric = imputeMedian(), factor = imputeConstant("_NA_")),
      dummy.classes = c("numeric", "integer")
    ), "ppc.scale" = TRUE, "ppc.center" = TRUE),
    adult, hout, measures = acc)

  # Plot PCA Projections
  data.frame(prcomp(embds$embed_occupation)$x) %>%
  mutate(name = rownames(.)) %>%
  ggplot() + geom_text(aes(x = PC1, y = PC2, label = name))

  # Learning Rate finder
  find_lr(mod, epochs = 10)

  # Tensorboard
  tensorboard(".")

  # --- 2071 Adult
  tsk = getOMLTask(2071)
  lrn = makeLearner("classif.embed_kerasff",
      "classif.embed_kerasff", lr = 0.1, epochs = 10,
      units_layer3 = 128, units_layer2 = 128, units_layer1 = 128,
      decay = 0.1, validation_split = 0)
  lrn = makePreprocWrapperCaret(
    makeImputeWrapper(lrn,
      classes = list(integer = imputeMedian(), numeric = imputeMedian(), factor = imputeConstant("_NA_")),
      dummy.classes = c("numeric", "integer")
    ), "ppc.scale" = TRUE, "ppc.center" = TRUE)
  runTaskMlr(tsk, lrn)

  # --- 145855 Ozone
  tsk = getOMLTask(145855)
  lrn = makeLearner("classif.embed_kerasff", lr = 0.01, mixup = TRUE, validation_split = 0)
  lrn = makePreprocWrapperCaret(
    makeImputeWrapper(lrn,
      classes = list(integer = imputeMedian(), numeric = imputeMedian(), factor = imputeConstant("_NA_")),
      dummy.classes = c("numeric", "integer")
    ), "ppc.scale" = TRUE, "ppc.center" = TRUE)
  runTaskMlr(tsk, lrn)
  # Default: 0.8543804
  # mixup:   0.9198895
  # bestcfg: 0.9463299
  # bestoml: 0.9499

  # Click-prediction small
  tsk = getOMLTask(14971)
  lrn = makeLearner("classif.embed_kerasff", validation_split = 0, lr = 10^-2, units_layer1 = 64, units_layer2 = 64, units_layer3 = 64)
  lrn = makePreprocWrapperCaret(
    makeImputeWrapper(lrn,
      classes = list(integer = imputeMedian(), numeric = imputeMedian(), factor = imputeConstant("_NA_")),
      dummy.classes = c("numeric", "integer")
    ), "ppc.scale" = TRUE, "ppc.center" = TRUE)
  # lr_finder(lrn, convertOMLTaskToMlr(tsk)$mlr.task, 10)
  runTaskMlr(tsk, lrn)
  # Default:    0.8319565
  # No valid:   0.8318063
  # bestcfg:    0.8318
  # bestoml:    0.8383
}


callback_wt_stats = callback_lambda(
    on_train_begin = function(logs) {wt_stats <<- c()},
    on_epoch_begin = function(batch, logs) {
      acts = c("leaky_re_lu", "leaky_re_lu_1", "leaky_re_lu_2")
      get_acts_funs = function(model, act) {
        btch = lapply(data$data, function(x) {
          if (is.null(dim(x))) z = x[seq_len(128)] else z = x[seq_len(128),]; return(z)
        })
        imod = keras_model(inputs = model$input, outputs = get_layer(model, act)$output)
        iout = predict(imod, btch)
        data.frame("mean" = apply(iout, 2, mean), "sd" = apply(iout, 2, sd), "layer" = act, "iter" = iter)
      }
      wt_stats <<- c(wt_stats, lapply(acts, get_acts_funs, model = model))
    }
)

