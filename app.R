library(shiny)
library(shinydashboard)
library(FactoMineR)
library(e1071)
library(randomForest)
library(nnet)
library(MASS)
library(caTools)
library(ggplot2)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(shinyWidgets)
library(plotly)
library(shinythemes)
library(promises)
library(future)
library(DT)
library(rmarkdown)
library(openxlsx)
library(blastula)
plan(multisession)

options(shiny.sanitize.errors = TRUE)

# Chemin vers les données par défaut (version Showcase)
default_data_path <- "data/Test_doc.csv"

ui <- dashboardPage(
  dashboardHeader(
    title = tagList(
      tags$img(src = "logo.png", height = "30px"), "MyDataBall - Showcase"
    )
  ),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Accueil", tabName = "home", icon = icon("home")),
      menuItem("Importation", tabName = "import", icon = icon("database")),
      menuItem("Nettoyage", tabName = "nettoyage", icon = icon("broom")),
      menuItem("Statistiques", tabName = "statistiques", icon = icon("chart-bar")),
      menuItem("Machine Learning", tabName = "ml", icon = icon("brain")),
      menuItem("Résultats & Reporting", tabName = "results", icon = icon("table")),
      menuItem("Aide", tabName = "help", icon = icon("question-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "home",
              fluidPage(
                box(
                  width = 12,
                  status = "primary",
                  h2("Bienvenue sur MyDataBall Showcase"),
                  p(
                    strong("Version interactive de démonstration"),
                    " – Version showcase de mon projet de stage chez MYDATABALL."
                  ),
                  p(
                    "Les données sont pré-chargées. Explorez import, nettoyage,",
                    " statistiques, ML, rapports et envoi par email."
                  ),
                  p(
                    "Aucun upload requis. Commencez par Importation,",
                    " puis naviguez entre les onglets."
                  ),
                  hr(),
                  h4("Fonctionnalités disponibles"),
                  tags$ul(
                    tags$li("Importation : Aperçu des données et résumé statistique"),
                    tags$li("Nettoyage : Options de préparation des données"),
                    tags$li("Statistiques : Analyses exploratoires et visualisations"),
                    tags$li("Machine Learning : Classification et régression"),
                    tags$li("Résultats & Reporting : Téléchargement et envoi par email")
                  )
                )
              )),
      tabItem(tabName = "import",
              fluidPage(
                h3("Importation des données"),
                p("Données chargées (jeu démo Test_doc.csv)."),
                textOutput("data_dimensions"),
                br(),
                h4("Aperçu des données"),
                DTOutput("data_preview"),
                br(),
                h4("Résumé statistique"),
                verbatimTextOutput("data_summary")
              )),
      tabItem(tabName = "nettoyage",
              fluidPage(
                h3("Nettoyage des données"),
                p(
                  "Options de préparation.",
                  " Modifications appliquées sur la copie de travail."
                ),
                fluidRow(
                  column(4,
                    checkboxInput("remove_na", "Supprimer les lignes avec NA", value = FALSE),
                    checkboxInput(
                      "remove_constant",
                      "Supprimer les colonnes à variance nulle",
                      value = FALSE
                    ),
                    actionButton(
                      "apply_cleaning", "Appliquer le nettoyage",
                      class = "btn btn-primary"
                    )
                  ),
                  column(8,
                    h4("Données après nettoyage"),
                    DTOutput("cleaned_data_preview")
                  )
                )
              )),
      tabItem(tabName = "statistiques",
              fluidPage(
                h3("Analyses statistiques"),
                fluidRow(
                  column(4,
                    selectInput("stat_var_x", "Variable X (numérique)", choices = NULL),
                    selectInput("stat_var_y", "Variable Y (numérique)", choices = NULL),
                    selectInput("stat_plot_type", "Type de graphique",
                      choices = c(
                        "Nuage de points" = "scatter",
                        "Histogramme" = "histogram",
                        "Barres" = "bar"
                      )
                    )
                  ),
                  column(8, plotlyOutput("stat_plot", height = "400px"))
                ),
                br(),
                h4("Résumé des variables numériques"),
                verbatimTextOutput("stat_summary")
              )),
      tabItem(tabName = "ml",
              fluidPage(
                h3("Machine Learning"),
                fluidRow(
                  column(6,
                    numericInput(
                      "num_targets", "Nombre de variables cibles",
                      value = 1, min = 1
                    ),
                    selectInput(
                      "analysis_type", "Type d'analyse",
                      choices = c("Classification", "Régression")
                    ),
                    uiOutput("analysis_choices_ui"),
                    numericInput(
                      "train_split",
                      "Pourcentage pour l'apprentissage",
                      value = 80, min = 50, max = 90
                    ),
                    actionButton(
                      "adjust_params", "Ajuster les paramètres",
                      class = "btn btn-primary"
                    ),
                    actionButton(
                      "run_analysis", "RUN", class = "btn btn-success"
                    ),
                    progressBar(id = "progress", value = 0, display_pct = TRUE)
                  )
                )
              )),
      tabItem(tabName = "results",
              fluidPage(
                h3("Résultats des analyses"),
                uiOutput("result_summaries"),
                plotlyOutput("roc_plot"),
                hr(),
                h4("Rapport et envoi par email"),
                p(
                  "Générez un rapport (PDF/Word), téléchargez-le ou envoyez-le.",
                  " Pour l'envoi : exécutez mail.blastula.R une fois pour configurer SMTP."
                ),
                fluidRow(
                  column(6, selectInput(
                    "result_format", "Format du rapport",
                    choices = c("PDF", "Word")
                  )),
                  column(6, textInput(
                    "email", "Email pour recevoir le rapport (optionnel)",
                    placeholder = "votre@email.com"
                  ))
                ),
                downloadButton(
                  "download_results", "Télécharger le rapport",
                  class = "btn btn-info"
                ),
                actionButton(
                  "send_email", "Envoyer par email",
                  class = "btn btn-primary"
                )
              )),
      tabItem(tabName = "help",
              fluidPage(
                h2("Aide"),
                p("Format CSV : séparateur ' ; ' et décimale ' . ' "),
                p("Les dernières colonnes sont réservées aux variables cibles à prédire."),
                p("Aucune donnée manquante n'est acceptée pour le Machine Learning."),
                p("Variables cibles : doivent être numériques (0, 1, 2, ...)."),
                p(
                  "Sur la version web, l'envoi par email peut être limité.",
                  " Pour un test complet, exécutez en local."
                ),
                br(),
                actionButton("user_guide_btn", "Guide Utilisateur"),
                actionButton("maintenance_guide_btn", "Guide de Maintenance")
              ))
    ),
    tags$div(
      id = "modals",
      modalDialog(
        id = "user_guide_modal", title = "Guide Utilisateur",
        easyClose = TRUE, footer = NULL
      ),
      modalDialog(
        id = "maintenance_guide_modal", title = "Guide de Maintenance",
        easyClose = TRUE, footer = NULL
      )
    )
  ),
  title = "MyDataBall - Showcase"
)

server <- function(input, output, session) {
  values <- reactiveValues(
    data = NULL,
    original_data = NULL
  )

  # Chargement automatique des données au démarrage (Showcase)
  load_default_data <- function() {
    path <- default_data_path
    if (!file.exists(path)) {
      path <- "Test_doc.csv"
    }
    df <- tryCatch({
      read.csv(path, header = TRUE, sep = ";", dec = ".")
    }, error = function(e) {
      showNotification(
        paste("Erreur lors du chargement :", e$message), type = "error"
      )
      return(NULL)
    })
    if (!is.null(df)) {
      values$data <- df
      values$original_data <- df
      showNotification(
        "Données chargées avec succès (version Showcase).", type = "message"
      )
    }
  }

  load_default_data()

  model_params <- reactiveValues(
    nn_layers = 1, nn_neurons = list(10), nn_activation = "relu",
    rf_trees = 100,
    logistic_threshold = 0.5, logistic_threshold_type = "alpha",
    tree_threshold = 0.5, tree_threshold_type = "alpha",
    linreg_targets = 1
  )
  analysis_results <- reactiveValues(
    train_results = list(),
    test_results = list(),
    train_roc = list(),
    test_roc = list(),
    train_r2 = list(),
    test_r2 = list()
  )

  # Dimensions des données
  output$data_dimensions <- renderText({
    req(values$data)
    paste0(
      "Nombre de lignes : ", nrow(values$data),
      ", Nombre de colonnes : ", ncol(values$data)
    )
  })

  # Aperçu des données (Importation)
  output$data_preview <- DT::renderDT({
    req(values$data)
    DT::datatable(
      values$data,
      options = list(pageLength = 10, scrollX = TRUE),
      rownames = FALSE
    )
  })

  # Résumé statistique (Importation)
  output$data_summary <- renderPrint({
    req(values$data)
    summary(values$data)
  })

  # Nettoyage
  observeEvent(input$apply_cleaning, {
    req(values$data)
    df <- values$data
    if (input$remove_na) {
      df <- df[complete.cases(df), ]
    }
    if (input$remove_constant) {
      df <- df[, sapply(df, function(col) {
        if (is.numeric(col)) length(unique(col)) > 1 else TRUE
      })]
    }
    values$data <- df
    showNotification("Nettoyage appliqué.", type = "message")
  })

  output$cleaned_data_preview <- DT::renderDT({
    req(values$data)
    DT::datatable(
      values$data,
      options = list(pageLength = 10, scrollX = TRUE),
      rownames = FALSE
    )
  })

  # Statistiques - variables numériques
  numeric_cols <- reactive({
    req(values$data)
    names(values$data)[sapply(values$data, is.numeric)]
  })

  observe({
    cols <- numeric_cols()
    if (length(cols) > 0) {
      updateSelectInput(session, "stat_var_x", choices = cols, selected = cols[1])
      if (length(cols) > 1) {
        sel_y <- cols[2]
      } else {
        sel_y <- cols[1]
      }
      updateSelectInput(session, "stat_var_y", choices = cols, selected = sel_y)
    }
  })

  output$stat_plot <- plotly::renderPlotly({
    req(values$data, input$stat_var_x)
    df <- values$data
    if (!input$stat_var_x %in% names(df)) {
      return(plotly::plot_ly())
    }
    x_vec <- df[[input$stat_var_x]]
    if (input$stat_plot_type == "scatter" && input$stat_var_y %in% names(df)) {
      plotly::plot_ly(
        df, x = x_vec, y = df[[input$stat_var_y]],
        type = "scatter", mode = "markers"
      ) %>%
        plotly::layout(
          xaxis = list(title = input$stat_var_x),
          yaxis = list(title = input$stat_var_y)
        )
    } else if (input$stat_plot_type == "histogram") {
      plotly::plot_ly(x = x_vec, type = "histogram") %>%
        plotly::layout(
          xaxis = list(title = input$stat_var_x),
          yaxis = list(title = "Fréquence")
        )
    } else if (input$stat_var_y %in% names(df)) {
      plotly::plot_ly(
        df, x = x_vec, y = df[[input$stat_var_y]], type = "bar"
      ) %>%
        plotly::layout(
          xaxis = list(title = input$stat_var_x),
          yaxis = list(title = input$stat_var_y)
        )
    } else {
      plotly::plot_ly(x = x_vec, type = "histogram") %>%
        plotly::layout(xaxis = list(title = input$stat_var_x))
    }
  })

  output$stat_summary <- renderPrint({
    req(values$data)
    summary(values$data[, numeric_cols(), drop = FALSE])
  })

  # Machine Learning
  observeEvent(input$analysis_type, {
    if (input$analysis_type == "Classification") {
      output$analysis_choices_ui <- renderUI({
        checkboxGroupInput(
          "analysis_choices", "Analyses de classification :",
          choices = c(
            "Régression logistique", "Réseaux de neurones",
            "Arbres de Décision", "Random Forest"
          ),
          selected = "Régression logistique"
        )
      })
    } else if (input$analysis_type == "Régression") {
      output$analysis_choices_ui <- renderUI({
        checkboxGroupInput(
          "analysis_choices", "Analyses de régression :",
          choices = c(
            "Régression linéaire", "Réseaux de neurones",
            "Random Forest", "Arbres de Décision"
          ),
          selected = "Régression linéaire"
        )
      })
    }
  })

  observeEvent(input$adjust_params, {
    showModal(modalDialog(
      title = "Ajuster les paramètres",
      tagList(
        if (length(input$analysis_choices) > 0 &&
            "Régression logistique" %in% input$analysis_choices) {
          tagList(
            h4("Régression logistique"),
            sliderInput(
              "logistic_threshold_modal", "Seuil",
              min = 0, max = 1,
              value = model_params$logistic_threshold, step = 0.01
            )
          )
        },
        if (length(input$analysis_choices) > 0 &&
            "Régression linéaire" %in% input$analysis_choices) {
          tagList(
            h4("Régression linéaire"),
            numericInput(
              "linreg_targets_modal", "Nombre de cibles",
              value = model_params$linreg_targets, min = 1
            )
          )
        },
        if (length(input$analysis_choices) > 0 &&
            "Réseaux de neurones" %in% input$analysis_choices) {
          tagList(
            h4("Réseaux de neurones"),
            numericInput(
              "nn_layers_modal", "Nombre de couches",
              value = model_params$nn_layers, min = 1
            ),
            uiOutput("nn_neurons_ui"),
            selectInput(
              "nn_activation_modal", "Fonction d'activation",
              choices = c("relu", "sigmoid", "tanh"),
              selected = model_params$nn_activation
            )
          )
        },
        if (length(input$analysis_choices) > 0 &&
            "Random Forest" %in% input$analysis_choices) {
          tagList(
            h4("Random Forest"),
            numericInput(
              "rf_trees_modal", "Nombre d'arbres",
              value = model_params$rf_trees, min = 1
            )
          )
        },
        if (length(input$analysis_choices) > 0 &&
            "Arbres de Décision" %in% input$analysis_choices) {
          tagList(
            h4("Arbres de Décision"),
            sliderInput(
              "tree_threshold_modal", "Seuil",
              min = 0, max = 1,
              value = model_params$tree_threshold, step = 0.01
            )
          )
        }
      ),
      footer = tagList(
        modalButton("Annuler"),
        actionButton("save_params", "Enregistrer")
      )
    ))
  })

  observeEvent(input$nn_layers_modal, {
    if (!is.null(input$nn_layers_modal)) {
      model_params$nn_layers <- input$nn_layers_modal
      output$nn_neurons_ui <- renderUI({
        lapply(1:model_params$nn_layers, function(i) {
          def_val <- ifelse(length(model_params$nn_neurons) >= i,
            model_params$nn_neurons[[i]], 10
          )
          numericInput(
            paste0("nn_neurons_", i), paste("Neurones couche", i),
            value = def_val, min = 1
          )
        })
      })
    }
  })

  observeEvent(input$save_params, {
    if (!is.null(input$logistic_threshold_modal)) {
      model_params$logistic_threshold <- input$logistic_threshold_modal
    }
    if (!is.null(input$linreg_targets_modal)) {
      model_params$linreg_targets <- input$linreg_targets_modal
    }
    if (!is.null(input$nn_layers_modal)) {
      model_params$nn_layers <- input$nn_layers_modal
    }
    if (!is.null(input$nn_activation_modal)) {
      model_params$nn_activation <- input$nn_activation_modal
    }
    if (!is.null(input$rf_trees_modal)) {
      model_params$rf_trees <- input$rf_trees_modal
    }
    if (!is.null(input$tree_threshold_modal)) {
      model_params$tree_threshold <- input$tree_threshold_modal
    }
    model_params$nn_neurons <- lapply(1:model_params$nn_layers, function(i) {
      input[[paste0("nn_neurons_", i)]]
    })
    removeModal()
  })

  run_analysis <- function(data_df, analysis_choices, num_targets,
                          model_params, session) {
    df <- data_df[, sapply(data_df, is.numeric)]
    df <- df[, sapply(df, function(col) length(unique(col)) > 1)]
    if (ncol(df) < num_targets + 1) {
      showNotification("Pas assez de colonnes numériques pour l'analyse.", type = "error")
      return(NULL)
    }
    target_cols <- tail(names(df), num_targets)
    feature_cols <- setdiff(names(df), target_cols)
    target_vector <- if (num_targets == 1) df[[target_cols]] else df[[target_cols]]
    data_preprocessed <- list(
      features = df[, feature_cols, drop = FALSE], targets = target_vector
    )
    set.seed(123)
    sample <- caTools::sample.split(
      data_preprocessed$targets, SplitRatio = input$train_split / 100
    )
    train_features <- data_preprocessed$features[sample == TRUE, , drop = FALSE]
    test_features <- data_preprocessed$features[sample == FALSE, , drop = FALSE]
    train_targets <- data_preprocessed$targets[sample == TRUE]
    test_targets <- data_preprocessed$targets[sample == FALSE]
    train_results <- list()
    test_results <- list()
    train_roc <- list()
    test_roc <- list()
    train_r2 <- list()
    test_r2 <- list()

    for (model in analysis_choices) {
      tryCatch({
        if (model == "Random Forest") {
          rf <- randomForest::randomForest(
            train_features, as.factor(train_targets), ntree = model_params$rf_trees
          )
          train_pred <- predict(rf, train_features, type = "response")
          test_pred <- predict(rf, test_features, type = "response")
          train_results[[model]] <- rf
          test_results[[model]] <- rf
          train_roc[[model]] <- pROC::roc(
            as.numeric(as.character(train_targets)), as.numeric(train_pred)
          )
          test_roc[[model]] <- pROC::roc(
            as.numeric(as.character(test_targets)), as.numeric(test_pred)
          )
          train_r2[[model]] <- tryCatch(
            caret::R2(train_pred, train_targets), error = function(e) NA
          )
          test_r2[[model]] <- tryCatch(
            caret::R2(test_pred, test_targets), error = function(e) NA
          )
        } else if (model == "Régression logistique") {
          n_classes <- length(unique(c(train_targets, test_targets)))
          if (n_classes > 2) {
            med <- median(as.numeric(unique(train_targets)))
            train_targets_bin <- as.integer(as.numeric(as.character(train_targets)) > med)
            test_targets_bin <- as.integer(as.numeric(as.character(test_targets)) > med)
          } else {
            train_targets_bin <- as.numeric(as.factor(train_targets)) - 1
            test_targets_bin <- as.numeric(as.factor(test_targets)) - 1
          }
          train_data <- data.frame(train_features, train_targets = train_targets_bin)
          glm_model <- glm(as.factor(train_targets) ~ ., data = train_data, family = binomial)
          train_pred <- predict(glm_model, newdata = train_data, type = "response")
          test_pred <- predict(
            glm_model, newdata = data.frame(test_features), type = "response"
          )
          train_targets <- train_targets_bin
          test_targets <- test_targets_bin
          train_results[[model]] <- glm_model
          test_results[[model]] <- glm_model
          train_roc[[model]] <- tryCatch(
            pROC::roc(train_targets, train_pred), error = function(e) NULL
          )
          test_roc[[model]] <- tryCatch(
            pROC::roc(test_targets, test_pred), error = function(e) NULL
          )
          train_r2[[model]] <- tryCatch(
            caret::R2(train_pred, train_targets), error = function(e) NA
          )
          test_r2[[model]] <- tryCatch(
            caret::R2(test_pred, test_targets), error = function(e) NA
          )
        } else if (model == "Réseaux de neurones") {
          nn_neurons <- unlist(model_params$nn_neurons)
          if (input$analysis_type == "Classification") {
            nn <- nnet(
              train_features, class.ind(as.factor(train_targets)),
              size = nn_neurons, linout = FALSE, softmax = TRUE,
              entropy = TRUE, maxit = 200
            )
            train_pred <- predict(nn, train_features, type = "class")
            test_pred <- predict(nn, test_features, type = "class")
          } else {
            nn <- nnet(
              train_features, train_targets, size = nn_neurons,
              linout = TRUE, decay = 0.1, maxit = 200
            )
            train_pred <- predict(nn, train_features, type = "raw")
            test_pred <- predict(nn, test_features, type = "raw")
          }
          train_results[[model]] <- nn
          test_results[[model]] <- nn
          train_r2[[model]] <- tryCatch(
            caret::R2(train_pred, train_targets), error = function(e) NA
          )
          test_r2[[model]] <- tryCatch(
            caret::R2(test_pred, test_targets), error = function(e) NA
          )
          if (input$analysis_type == "Classification") {
            train_roc[[model]] <- pROC::roc(
              as.numeric(as.character(train_targets)), as.numeric(train_pred)
            )
            test_roc[[model]] <- pROC::roc(
              as.numeric(as.character(test_targets)), as.numeric(test_pred)
            )
          }
        } else if (model == "Arbres de Décision") {
          train_df <- data.frame(train_features, train_targets)
          tree_model <- rpart(
            as.factor(train_targets) ~ ., data = train_df, method = "class"
          )
          train_pred <- predict(
            tree_model, data.frame(train_features), type = "prob"
          )[, 2]
          test_pred <- predict(
            tree_model, data.frame(test_features), type = "prob"
          )[, 2]
          train_results[[model]] <- tree_model
          test_results[[model]] <- tree_model
          train_roc[[model]] <- pROC::roc(
            as.numeric(as.character(train_targets)), train_pred
          )
          test_roc[[model]] <- pROC::roc(
            as.numeric(as.character(test_targets)), test_pred
          )
          train_r2[[model]] <- tryCatch(
            caret::R2(train_pred, train_targets), error = function(e) NA
          )
          test_r2[[model]] <- tryCatch(
            caret::R2(test_pred, test_targets), error = function(e) NA
          )
        } else if (model == "Régression linéaire") {
          lin_df <- data.frame(train_targets = train_targets, train_features)
          linreg <- lm(train_targets ~ ., data = lin_df)
          train_pred <- predict(linreg, newdata = data.frame(train_features))
          test_pred <- predict(linreg, newdata = data.frame(test_features))
          train_results[[model]] <- linreg
          test_results[[model]] <- linreg
          train_r2[[model]] <- caret::R2(train_pred, train_targets)
          test_r2[[model]] <- caret::R2(test_pred, test_targets)
        }
      }, error = function(e) {
        showNotification(paste("Erreur modèle", model, ":", e$message), type = "error")
      })
    }
    list(
      train_results = train_results, test_results = test_results,
      train_roc = train_roc, test_roc = test_roc,
      train_r2 = train_r2, test_r2 = test_r2
    )
  }

  observeEvent(input$run_analysis, {
    req(values$data, input$analysis_choices)
    df <- values$data
    if (anyNA(df)) {
      showNotification(
        "Les données contiennent des NA. Utilisez l'onglet Nettoyage.",
        type = "error"
      )
      return()
    }
    results <- run_analysis(
      df, input$analysis_choices, input$num_targets, model_params, session
    )
    if (is.null(results)) {
      return()
    }
    analysis_results$train_results <- results$train_results
    analysis_results$test_results <- results$test_results
    analysis_results$train_roc <- results$train_roc
    analysis_results$test_roc <- results$test_roc
    analysis_results$train_r2 <- results$train_r2
    analysis_results$test_r2 <- results$test_r2
    showNotification("Analyse terminée.", type = "message")
  })

  output$result_summaries <- renderUI({
    if (length(analysis_results$train_results) == 0) {
      return(p("Lancez une analyse depuis l'onglet Machine Learning."))
    }
    summaries <- lapply(seq_along(names(analysis_results$train_results)), function(i) {
      model_name <- names(analysis_results$train_results)[i]
      model_id <- paste0("m", i)
      train_result <- analysis_results$train_results[[model_name]]
      train_r2_val <- analysis_results$train_r2[[model_name]]
      if (!is.null(train_r2_val)) {
        train_r2_value <- round(train_r2_val, 2)
      } else {
        train_r2_value <- NA
      }
      test_r2_val <- analysis_results$test_r2[[model_name]]
      if (!is.null(test_r2_val)) {
        test_r2_value <- round(test_r2_val, 2)
      } else {
        test_r2_value <- NA
      }
      train_auc_obj <- analysis_results$train_roc[[model_name]]
      if (!is.null(train_auc_obj)) {
        train_auc_value <- round(pROC::auc(train_auc_obj), 2)
      } else {
        train_auc_value <- NA
      }
      test_auc_obj <- analysis_results$test_roc[[model_name]]
      if (!is.null(test_auc_obj)) {
        test_auc_value <- round(pROC::auc(test_auc_obj), 2)
      } else {
        test_auc_value <- NA
      }
      fluidRow(
        box(width = 12, title = model_name, status = "info",
            fluidRow(
              column(6, h5("Apprentissage"),
                p("R²: ", train_r2_value), p("AUC: ", train_auc_value)),
              column(6, h5("Test"),
                p("R²: ", test_r2_value), p("AUC: ", test_auc_value))
            ),
            if (inherits(train_result, "randomForest")) {
              plotOutput(paste0("train_importance_plot_", model_id))
            } else if (inherits(train_result, "glm") || inherits(train_result, "lm")) {
              verbatimTextOutput(paste0("train_summary_", model_id))
            } else if (inherits(train_result, "nnet")) {
              verbatimTextOutput(paste0("train_summary_", model_id))
            } else if (inherits(train_result, "rpart")) {
              plotOutput(paste0("train_tree_plot_", model_id))
            }
        )
      )
    })
    do.call(tagList, summaries)
  })

  output$roc_plot <- plotly::renderPlotly({
    if (length(analysis_results$train_roc) == 0 || length(analysis_results$test_roc) == 0) {
      return(plotly::plot_ly())
    }
    valid_models <- names(analysis_results$train_roc)[
      sapply(analysis_results$train_roc, function(x) !is.null(x))
    ]
    if (length(valid_models) == 0) {
      return(plotly::plot_ly())
    }
    plots <- lapply(valid_models, function(model_name) {
      tr <- analysis_results$train_roc[[model_name]]
      te <- analysis_results$test_roc[[model_name]]
      if (is.null(tr) || is.null(te)) return(NULL)
      plotly::plot_ly(
        x = tr$specificities, y = tr$sensitivities,
        type = "scatter", mode = "lines", name = paste(model_name, "- Train")
      ) %>%
        plotly::add_trace(
          x = te$specificities, y = te$sensitivities,
          type = "scatter", mode = "lines", name = paste(model_name, "- Test")
        ) %>%
        plotly::layout(
          title = paste("ROC -", model_name),
          xaxis = list(title = "1 - Specificity"),
          yaxis = list(title = "Sensitivity")
        )
    })
    plots <- plots[!sapply(plots, is.null)]
    if (length(plots) > 0) {
      plotly::subplot(
        plots, nrows = ceiling(length(plots) / 2),
        shareX = TRUE, shareY = TRUE
      )
    } else {
      plotly::plot_ly()
    }
  })

  observe({
    model_names <- names(analysis_results$train_results)
    for (i in seq_along(model_names)) {
      local({
        ii <- i
        model_id <- paste0("m", ii)
        model_name <- model_names[ii]
        train_result <- analysis_results$train_results[[model_name]]
        if (inherits(train_result, "randomForest")) {
          output[[paste0("train_importance_plot_", model_id)]] <- renderPlot({
            randomForest::varImpPlot(analysis_results$train_results[[model_name]])
          })
        } else if (inherits(train_result, "glm") || inherits(train_result, "lm")) {
          output[[paste0("train_summary_", model_id)]] <- renderPrint({
            summary(analysis_results$train_results[[model_name]])
          })
        } else if (inherits(train_result, "nnet")) {
          output[[paste0("train_summary_", model_id)]] <- renderPrint({
            summary(analysis_results$train_results[[model_name]])
          })
        } else if (inherits(train_result, "rpart")) {
          output[[paste0("train_tree_plot_", model_id)]] <- renderPlot({
            rpart.plot::rpart.plot(analysis_results$train_results[[model_name]])
          })
        }
      })
    }
  })

  output$download_results <- downloadHandler(
    filename = function() {
      fmt <- switch(input$result_format, "PDF" = ".pdf", "Word" = ".docx")
      paste0("rapport_mydataball_", Sys.Date(), fmt)
    },
    content = function(file) {
      temp_report <- file.path(tempdir(), "report_template.Rmd")
      template_path <- "report_template.Rmd"
      if (!file.exists(template_path)) template_path <- file.path("..", "report_template.Rmd")
      file.copy(template_path, temp_report, overwrite = TRUE)
      params <- list(
        train_results = analysis_results$train_results,
        test_results = analysis_results$test_results,
        train_r2_values = analysis_results$train_r2,
        test_r2_values = analysis_results$test_r2,
        train_auc_values = lapply(analysis_results$train_roc, function(x) {
          if (!is.null(x)) pROC::auc(x) else NA
        }),
        test_auc_values = lapply(analysis_results$test_roc, function(x) {
          if (!is.null(x)) pROC::auc(x) else NA
        }),
        train_roc_plots = analysis_results$train_roc,
        test_roc_plots = analysis_results$test_roc,
        model_summaries = lapply(analysis_results$train_results, summary)
      )
      if (input$result_format == "PDF") {
        rmarkdown::render(
          temp_report, output_file = file, params = params,
          envir = new.env(parent = globalenv()),
          output_format = "pdf_document"
        )
      } else {
        rmarkdown::render(
          temp_report, output_file = file, params = params,
          envir = new.env(parent = globalenv()),
          output_format = "word_document"
        )
      }
    }
  )

  observeEvent(input$send_email, {
    if (is.null(input$email) || nchar(trimws(input$email)) == 0) {
      showNotification("Veuillez saisir une adresse email.", type = "warning")
      return()
    }
    if (length(analysis_results$train_results) == 0) {
      showNotification(
        "Lancez une analyse Machine Learning avant d'envoyer le rapport.",
        type = "warning"
      )
      return()
    }
    showNotification("Génération du rapport en cours...", type = "message")

    template_path <- "report_template.Rmd"
    if (!file.exists(template_path)) {
      template_path <- file.path("..", "report_template.Rmd")
    }
    if (!file.exists(template_path)) {
      showNotification(
        "Fichier report_template.Rmd introuvable.",
        type = "error"
      )
      return()
    }

    temp_report <- file.path(tempdir(), "report_template.Rmd")
    file.copy(template_path, temp_report, overwrite = TRUE)
    params <- list(
      train_results = analysis_results$train_results,
      test_results = analysis_results$test_results,
      train_r2_values = analysis_results$train_r2,
      test_r2_values = analysis_results$test_r2,
      train_auc_values = lapply(analysis_results$train_roc, function(x) {
        if (!is.null(x)) pROC::auc(x) else NA
      }),
      test_auc_values = lapply(analysis_results$test_roc, function(x) {
        if (!is.null(x)) pROC::auc(x) else NA
      }),
      train_roc_plots = analysis_results$train_roc,
      test_roc_plots = analysis_results$test_roc,
      model_summaries = lapply(analysis_results$train_results, summary)
    )
    ext <- switch(input$result_format, "PDF" = ".pdf", "Word" = ".docx")
    temp_file <- tempfile(fileext = ext)

    tryCatch({
      if (input$result_format == "PDF") {
        rmarkdown::render(
          temp_report, output_file = temp_file, params = params,
          envir = new.env(parent = globalenv()),
          output_format = "pdf_document"
        )
      } else {
        rmarkdown::render(
          temp_report, output_file = temp_file, params = params,
          envir = new.env(parent = globalenv()),
          output_format = "word_document"
        )
      }

      email <- blastula::compose_email(
        body = blastula::md("Voici le rapport généré par MyDataBall Showcase."),
        footer = blastula::md("Projet portfolio - MyDataBall")
      ) %>% blastula::add_attachment(temp_file)

      creds <- tryCatch(
        blastula::creds_key("my_smtp_key"),
        error = function(e) NULL
      )
      if (is.null(creds)) {
        showNotification(
          paste(
            "Clé SMTP non configurée. Exécutez mail.blastula.R dans R,",
            "puis entrez le mot de passe (App Password Gmail si 2FA)."
          ),
          type = "error", duration = 10
        )
        return()
      }

      blastula::smtp_send(
        email,
        from = "bendabire107@gmail.com",
        to = trimws(input$email),
        subject = "Rapport MyDataBall Showcase",
        credentials = creds
      )
      showNotification("Email envoyé avec succès.", type = "message")
    }, error = function(e) {
      err_msg <- as.character(e$message)
      if (grepl("credential|key|keyring", err_msg, ignore.case = TRUE)) {
        showNotification(
          paste(
            "Configuration SMTP manquante. Exécutez mail.blastula.R hors Shiny,",
            "puis créez la clé my_smtp_key avec votre mot de passe Gmail",
            "(ou App Password si 2FA activée)."
          ),
          type = "error", duration = 10
        )
      } else if (grepl("Login denied|authentication|535|534", err_msg)) {
        showNotification(
          paste(
            "Login refusé par Gmail. Créez un Mot de passe d'application :",
            "google.com/account-security > Mots de passe d'application.",
            "Puis exécutez mail.blastula.R avec overwrite=TRUE."
          ),
          type = "error", duration = 12
        )
      } else {
        showNotification(
          paste("Erreur :", err_msg),
          type = "error", duration = 8
        )
      }
    })
  })

  observeEvent(input$user_guide_btn, {
    showModal(modalDialog(
      title = "Guide Utilisateur",
      tagList(
        p("Interface Shiny MyDataBall Showcase - Version démonstration."),
        p(
          "Données pré-chargées. Importation, Nettoyage, Statistiques,",
          " ML, puis Résultats pour télécharger ou envoyer le rapport."
        )
      ),
      easyClose = TRUE, footer = NULL
    ))
  })

  observeEvent(input$maintenance_guide_btn, {
    showModal(modalDialog(
      title = "Guide de Maintenance",
      tagList(
        p(
          "Configuration SMTP : exécuter mail.blastula.R.",
          " Template rapport : report_template.Rmd."
        )
      ),
      easyClose = TRUE, footer = NULL
    ))
  })
}

shinyApp(ui, server)
