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
      makeIntegerLearnerParam(id = "layers", lower = 1L, upper = 4L, default = 1L),
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
  epochs = 30L, early_stopping_patience = 5L, learning_rate_scheduler = FALSE,
  optimizer = "adam", lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, momentum = 0, decay = 0,
  rho = 0.9, loss = "categorical_crossentropy", batch_size = 128L, n_layers = 1,
  batchnorm_dropout = "dropout", input_dropout_rate = 0, dropout_rate = 0,
  units_layer1 = 32, units_layer2 = 32, units_layer3 = 32, units_layer4 = 32, init_layer = "glorot_uniform",
  act_layer = "relu", l1_reg_layer = 0.01, l2_reg_layer = 0.01, validation_split = 0.2) {

  require("keras")
  input_shape = getTaskNFeats(.task)
  output_shape = length(getTaskClassLevels(.task))
  data = getTaskData(.task, .subset, target.extra = TRUE)


  # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
  # Dense -> Act -> [BN] -> [Dropout]
  regularizer = regularizer_l1_l2(l1 = l1_reg_layer, l2 = l2_reg_layer)
  optimizer = switch(optimizer,
    "sgd" = optimizer_sgd(lr, momentum, decay = decay),
    "rmsprop" = optimizer_rmsprop(lr, rho, decay = decay),
    "adagrad" = optimizer_adagrad(lr, decay = decay),
    "adam" = optimizer_adam(lr, beta_1, beta_2, decay = decay),
    "nadam" = optimizer_nadam(lr, beta_1, beta_2, schedule_decay = decay)
  )

  callbacks = c()
  if (early_stopping_patience > 0)
    callbacks = c(callbacks, callback_early_stopping(monitor = 'val_loss', patience = early_stopping_patience))
  if (learning_rate_scheduler)
    callbacks = c(callback_learning_rate_scheduler(function(epoch, lr) {lr * 1/(1 * epoch)}))


  # --- Build Up Model -------------------------------------------------------
  units_layers = c(units_layer1, units_layer2, units_layer3, units_layer4)

  # The model consists of Embedding layers for the categorical variables,
  # followed by a Dropout of emb_drop, and a BatchNorm for the continuous variables.
  # The results are concatenated and followed by blocks of BatchNorm, Dropout, 
  # Linear and ReLU (the first block skips BatchNorm and Dropout, the last block skips
  # the ReLU).
  embedding = make_embedding(data$data)
  layers = embedding$layers
  for (i in seq_len(n_layers + 1L)) {
    if (i > 1) {
      layers = layers %>%
        layer_batch_normalization() %>%
        layer_dropout(dropout_rate)
    }
    # Final layer 
    if (i != n_layers)
    layers = layers %>% layer_dense(units = units_layers[i]) %>%
      layer_activation_relu()
    else 
      layers = layers %>% layer_dense(units = output_shape, activation = "softmax")
  }

  model = keras_model(input = embedding$inputs, output = layers)

  # --- Compile and Fit ---------------------------------------------------------------
  model %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = c('accuracy')
  )
  data = reshape_data_embedding(data$data, data$target)
  model %>% fit(
    x = data$data,
    y = to_categorical(data$label, num_classes = output_shape),
    epochs = epochs,
    batch_size = 32L,
    validation_split = validation_split,
    callbacks = callbacks
  )

  return(list(model = model, history = history, target_levels = levels(data$target)))
}

predictLearner.classif.embed_kerasff = function(.learner, .model, .newdata, ...) {
  if (.learner$predict.type == "prob") {
    p = .model$learner.model$model %>% predict_proba(as.matrix(.newdata))
    colnames(p) = .model$learner.model$target_levels
  } else {
    p = .model$learner.model$model %>% predict_classes(as.matrix(.newdata))
    labels = .model$learner.model$target_levels[unique(p + 1)]
    p = factor(p, labels = labels)
  }
  return(p)
}

reshape_data_embedding = function(data, target) {
  assert_factor(target)
  assert_data_frame(data)
  type = BBmisc::vcapply(data, function(x) class(x)[[1]])
  embed_vars = type %in% c("ordered", "factor")

  out_data = list()
  if (sum(embed_vars)  > 0)
    out_data = setNames(lapply(as.list(data[, embed_vars]), function(x) as.integer(x) - 1L), colnames(data)[embed_vars])
  if (sum(!embed_vars) > 0)
    out_data$continuous = BBmisc::normalize(as.matrix(data[, !embed_vars]), margin = 2L)

  list(
    data = out_data,
    label =  array(as.integer(target) - 1, dim = c(nrow(data), 1))
  )
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
      layer_embedding(input_dim = as.numeric(n_cat), output_dim = as.numeric(embed_size), input_length = 1) %>%
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


if (FALSE) {
  lrn = makeLearner("classif.embed_kerasff")
  train(lrn, iris.task)
}