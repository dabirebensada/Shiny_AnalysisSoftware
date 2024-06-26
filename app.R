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

ui <- dashboardPage(
  dashboardHeader(title = tagList(tags$img(src = "logo.png", height = "30px"), "Interface d'Analyse")),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Accueil", tabName = "home", icon = icon("home")),
      menuItem("Upload", tabName = "upload", icon = icon("upload")),
      menuItem("Résultats", tabName = "results", icon = icon("table")),
      menuItem("Aide", tabName = "help", icon = icon("question-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "home",
              fluidPage(
                h2("BIENVENUE"),
                p("Cette interface vous permet de réaliser diverses analyses de classification ou de régression."),
                p("L'application est conçue pour fonctionner avec n'importe quelle base de données au format CSV, tant qu'elle respecte un certain nombre de critères."),
                p("Pour plus de précision et pour les nouveaux utilisateurs, veuillez lire attentivement la section Aide afin de prendre connaissance des conditions d'utilisation des différents modèles disponibles.")
              )),
      tabItem(tabName = "upload",
              fluidPage(
                fileInput("datafile", "Choisir un fichier CSV", accept = ".csv"),
                textOutput("data_dimensions"),  
                numericInput("num_targets", "Nombre de variables cibles", value = 1, min = 1),
                selectInput("analysis_type", "Type d'analyse", choices = c("Classification", "Régression")),
                uiOutput("analysis_choices_ui"),
                numericInput("train_split", "Pourcentage de données pour l'apprentissage", value = 80, min = 50, max = 90),
                actionButton("adjust_params", "Ajuster les paramètres", class = "btn btn-primary"),
                actionButton("run_analysis", "RUN", class = "btn btn-success"),
                progressBar(id = "progress", value = 0, display_pct = TRUE)
              )),
      tabItem(tabName = "results",
              fluidPage(
                h3("Résultats des Analyses"),
                uiOutput("result_summaries"),
                plotlyOutput("roc_plot"),
                fluidRow(
                  column(6, selectInput("result_format", "Choisir le format de téléchargement", choices = c("PDF", "Word"))),
                  column(6, textInput("email", "Email pour envoyer les résultats (optionnel)"))
                ),
                downloadButton("download_results", "Télécharger les résultats", class = "btn btn-info"),
                actionButton("send_email", "Envoyer par email", class = "btn btn-primary")
              )),
      tabItem(tabName = "help",
              fluidPage(
                h2("Aide"),
                p("Format CSV : séparateur ' ; ' et décimale ' . ' "),
                p("Les dernières colonnes sont réservées aux variables cibles à prédire."),
                p("Aucune donnée manquante n'est acceptée (nous consulter sur les choix et la stratégie pour compléter les données)."),
                p("Si une variable n'a qu'un seul attribut, elle sera éliminée (nombre de valeurs distinctes > 1)."),
                p("Ne pas mettre de variables INDEX (c'est une variable numérique ou symbolique qui a autant d'attributs qu'il n'y a de lignes)."),
                p("Variables cibles : doivent être numériques (0, 1, 2, ...), integer ou réel (pour une classification elles doivent être impérativement en integer)."),
                p("Les méta données donnent le type de variable :"),
                p("- symboliques (string) = vérifiez que les nomenclatures sont cohérentes et que pour une même valeur, elles sont écrites de la même façon (exemple 'NA' != 'na') ; on élimine les variables qui ont autant de valeurs qu'il y a de lignes ;"),
                p("- numériques (hexadécimales) = si vous considérez que des variables doivent être normalisées pour mieux les comparer, faire cette manipulation en amont de l'utilisation du logiciel (par exemple : faire une normalisation centrée réduite). "),
                br(),
                actionButton("user_guide_btn", "Guide Utilisateur"),
                actionButton("maintenance_guide_btn", "Guide de Maintenance")
              ))
    )
  ),
  
  # Modals pour les guides
  tags$div(id = 'modals',
           modalDialog(id = 'user_guide_modal', title = "Guide Utilisateur", easyClose = TRUE, footer = NULL),
           modalDialog(id = 'maintenance_guide_modal', title = "Guide de Maintenance", easyClose = TRUE, footer = NULL)
  )
)

server <- function(input, output, session) {
  data <- reactiveVal(NULL)
  model_params <- reactiveValues(
    nn_layers = 1, nn_neurons = list(), nn_activation = "relu",
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
  
  observeEvent(input$analysis_type, {
    if (input$analysis_type == "Classification") {
      output$analysis_choices_ui <- renderUI({
        checkboxGroupInput("analysis_choices", "Choisissez les analyses de classification :",
                           choices = c("Régression logistique", "Réseaux de neurones", "Arbres de Décision", "Random Forest"),
                           selected = "Régression logistique")
      })
    } else if (input$analysis_type == "Régression") {
      output$analysis_choices_ui <- renderUI({
        checkboxGroupInput("analysis_choices", "Choisissez les analyses de régression :",
                           choices = c("Régression linéaire", "Régression logistique", "Réseaux de neurones", "Random Forest", "Arbres de Décision"),
                           selected = "Régression linéaire")
      })
    }
  })
  
  observeEvent(input$datafile, {
    req(input$datafile)
    file <- input$datafile$datapath
    df <- tryCatch({
      read.csv(file, sep = ";", dec = ".", header = TRUE)
    }, error = function(e) {
      showNotification(paste("Erreur lors du chargement du fichier CSV :", e$message), type = "error")
      return(NULL)
    })
    
    # Vérification des en-têtes
    if (ncol(df) < input$num_targets + 1) {
      showNotification("Le fichier CSV doit contenir au moins autant de colonnes +1 que le nombre de variables cibles spécifiées.", type = "error")
      return(NULL)
    }
    
    # Vérification des valeurs manquantes
    if (anyNA(df)) {
      showNotification("Les données contiennent des valeurs manquantes. Veuillez les compléter avant de continuer.", type = "error")
      return(NULL)
    }
    
    # Vérification des noms des colonnes
    if (any(grepl("\\s|[^[:alnum:]_]", names(df)))) {
      showNotification("Les noms des colonnes ne doivent pas contenir d'espaces ou de caractères spéciaux.", type = "error")
      return(NULL)
    }
    
    data(df)
    showNotification("Les données ont été chargées avec succès.", type = "message")
    
    # Mettre à jour les dimensions des données
    output$data_dimensions <- renderText({
      paste("Nombre de lignes : ", nrow(df), ", Nombre de colonnes : ", ncol(df))
    })
  })
  
  observeEvent(input$adjust_params, {
    showModal(modalDialog(
      title = "Ajuster les paramètres",
      tagList(
        if ("Régression logistique" %in% input$analysis_choices) {
          tagList(
            h4("Régression logistique"),
            sliderInput("logistic_threshold_modal", "Seuil", min = 0, max = 1, value = model_params$logistic_threshold, step = 0.01)
          )
        },
        if ("Régression linéaire" %in% input$analysis_choices) {
          tagList(
            h4("Régression linéaire"),
            numericInput("linreg_targets_modal", "Nombre de cibles", value = model_params$linreg_targets, min = 1)
          )
        },
        if ("Réseaux de neurones" %in% input$analysis_choices) {
          tagList(
            h4("Réseaux de neurones"),
            numericInput("nn_layers_modal", "Nombre de couches", value = model_params$nn_layers, min = 1),
            uiOutput("nn_neurons_ui"),
            selectInput("nn_activation_modal", "Fonction d'activation", choices = c("relu", "sigmoid", "tanh"), selected = model_params$nn_activation)
          )
        },
        if ("Random Forest" %in% input$analysis_choices) {
          tagList(
            h4("Random Forest"),
            numericInput("rf_trees_modal", "Nombre d'arbres", value = model_params$rf_trees, min = 1)
          )
        },
        if ("Arbres de Décision" %in% input$analysis_choices) {
          tagList(
            h4("Arbres de Décision"),
            sliderInput("tree_threshold_modal", "Seuil", min = 0, max = 1, value = model_params$tree_threshold, step = 0.01)
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
    model_params$nn_layers <- input$nn_layers_modal
    output$nn_neurons_ui <- renderUI({
      lapply(1:model_params$nn_layers, function(i) {
        numericInput(paste0("nn_neurons_", i), paste("Nombre de neurones pour la couche", i), value = ifelse(length(model_params$nn_neurons) >= i, model_params$nn_neurons[[i]], 10), min = 1)
      })
    })
  })
  
  observeEvent(input$save_params, {
    if (!is.null(input$logistic_threshold_modal)) model_params$logistic_threshold <- input$logistic_threshold_modal
    if (!is.null(input$linreg_targets_modal)) model_params$linreg_targets <- input$linreg_targets_modal
    if (!is.null(input$nn_layers_modal)) model_params$nn_layers <- input$nn_layers_modal
    if (!is.null(input$nn_activation_modal)) model_params$nn_activation <- input$nn_activation_modal
    if (!is.null(input$rf_trees_modal)) model_params$rf_trees <- input$rf_trees_modal
    if (!is.null(input$tree_threshold_modal)) model_params$tree_threshold <- input$tree_threshold_modal
    
    model_params$nn_neurons <- lapply(1:model_params$nn_layers, function(i) {
      input[[paste0("nn_neurons_", i)]]
    })
    
    removeModal()
  })
  
  run_analysis <- function(data, analysis_choices, num_targets, model_params, session) {
    # Suppression des colonnes non numériques et des colonnes avec une seule valeur distincte
    df <- data[, sapply(data, is.numeric)]
    df <- df[, sapply(df, function(col) length(unique(col)) > 1)]
    
    # Extraction des cibles
    target_cols <- tail(names(df), num_targets)
    feature_cols <- setdiff(names(df), target_cols)
    
    # Vérification que les données cibles sont bien des vecteurs
    if (num_targets == 1) {
      target_vector <- df[[target_cols]]
    } else {
      target_vector <- df[target_cols]
    }
    
    # Sauvegarde des données prétraitées dans une variable réactive
    data_preprocessed <- list(features = df[, feature_cols], targets = target_vector)
    
    # Séparation des données en ensembles d'entraînement et de test
    set.seed(123)
    sample <- sample.split(data_preprocessed$targets, SplitRatio = input$train_split / 100)
    
    train_features <- data_preprocessed$features[sample == TRUE, ]
    test_features <- data_preprocessed$features[sample == FALSE, ]
    train_targets <- data_preprocessed$targets[sample == TRUE]
    test_targets <- data_preprocessed$targets[sample == FALSE]
    
    # Initialisation des listes de résultats
    train_results <- list()
    test_results <- list()
    train_roc <- list()
    test_roc <- list()
    train_r2 <- list()
    test_r2 <- list()
    
    # Exécution de chaque modèle sélectionné
    for (model in analysis_choices) {
      tryCatch({
        if (model == "Random Forest") {
          rf <- randomForest(train_features, as.factor(train_targets), ntree = model_params$rf_trees)
          train_pred <- predict(rf, train_features, type = "response")
          test_pred <- predict(rf, test_features, type = "response")
          
          train_results[[model]] <- rf
          test_results[[model]] <- rf
          train_roc[[model]] <- roc(train_targets, as.numeric(train_pred))
          test_roc[[model]] <- roc(test_targets, as.numeric(test_pred))
          train_r2[[model]] <- R2(train_pred, train_targets)
          test_r2[[model]] <- R2(test_pred, test_targets)
        } else if (model == "Régression logistique") {
          # Créez un data frame avec les caractéristiques et les cibles
          train_data <- data.frame(train_features, train_targets)
          test_data <- data.frame(test_features)
          colnames(train_data)[ncol(train_data)] <- "train_targets"
          
          # Ajustez le modèle de régression logistique
          glm_model <- glm(as.factor(train_targets) ~ ., data = train_data, family = binomial)
          
          # Prédictions sur les données d'entraînement et de test
          train_pred <- predict(glm_model, newdata = train_data, type = "response")
          test_pred <- predict(glm_model, newdata = test_data, type = "response")
          
          # Stockez les résultats
          train_results[[model]] <- glm_model
          test_results[[model]] <- glm_model
          train_roc[[model]] <- roc(train_targets, train_pred)
          test_roc[[model]] <- roc(test_targets, test_pred)
          train_r2[[model]] <- R2(train_pred, train_targets)
          test_r2[[model]] <- R2(test_pred, test_targets)
        } else if (model == "Réseaux de neurones") {
          nn_layers <- model_params$nn_layers
          nn_neurons <- unlist(model_params$nn_neurons)
          
          if (input$analysis_type == "Classification") {
            nn <- nnet(train_features, class.ind(as.factor(train_targets)), size = nn_neurons, linout = FALSE, softmax = TRUE, entropy = TRUE, maxit = 200)
          } else {
            nn <- nnet(train_features, train_targets, size = nn_neurons, linout = TRUE, decay = 0.1, maxit = 200)
          }
          
          train_pred <- predict(nn, train_features, type = "class")
          test_pred <- predict(nn, test_features, type = "class")
          
          train_results[[model]] <- nn
          test_results[[model]] <- nn
          train_roc[[model]] <- roc(train_targets, train_pred)
          test_roc[[model]] <- roc(test_targets, test_pred)
          train_r2[[model]] <- R2(train_pred, train_targets)
          test_r2[[model]] <- R2(test_pred, test_targets)
        } else if (model == "Arbres de Décision") {
          tree_model <- rpart(as.factor(train_targets) ~ ., data = data.frame(train_features, train_targets), method = "class")
          train_pred <- predict(tree_model, data.frame(train_features), type = "prob")[, 2]
          test_pred <- predict(tree_model, data.frame(test_features), type = "prob")[, 2]
          
          train_results[[model]] <- tree_model
          test_results[[model]] <- tree_model
          train_roc[[model]] <- roc(train_targets, train_pred)
          test_roc[[model]] <- roc(test_targets, test_pred)
          train_r2[[model]] <- R2(train_pred, train_targets)
          test_r2[[model]] <- R2(test_pred, test_targets)
        } else if (model == "Régression linéaire") {
          # Vérifier et renommer la colonne cible pour éviter les conflits
          colnames(train_features)[colnames(train_features) == target_cols[1]] <- "y"
          colnames(test_features)[colnames(test_features) == target_cols[1]] <- "y"
          
          linreg <- lm(y ~ ., data = data.frame(y = train_targets, train_features))
          train_pred <- predict(linreg, newdata = data.frame(train_features))
          test_pred <- predict(linreg, newdata = data.frame(test_features))
          
          train_results[[model]] <- linreg
          test_results[[model]] <- linreg
          train_r2[[model]] <- R2(train_pred, train_targets)
          test_r2[[model]] <- R2(test_pred, test_targets)
        }
      }, error = function(e) {
        showNotification(paste("Erreur lors de l'exécution du modèle", model, ":", e$message), type = "error")
      })
    }
    
    return(list(train_results = train_results, test_results = test_results, train_roc = train_roc, test_roc = test_roc, train_r2 = train_r2, test_r2 = test_r2, train_features = train_features, test_features = test_features, train_targets = train_targets, test_targets = test_targets))
  }
  
  observeEvent(input$run_analysis, {
    req(data(), input$analysis_choices)
    
    analysis_choices <- input$analysis_choices
    num_targets <- input$num_targets
    df <- data()
    
    results <- run_analysis(df, analysis_choices, num_targets, model_params, session)
    analysis_results$train_results <- results$train_results
    analysis_results$test_results <- results$test_results
    analysis_results$train_roc <- results$train_roc
    analysis_results$test_roc <- results$test_roc
    analysis_results$train_r2 <- results$train_r2
    analysis_results$test_r2 <- results$test_r2
    
    output$result_summaries <- renderUI({
      summaries <- lapply(names(analysis_results$train_results), function(model_name) {
        train_result <- analysis_results$train_results[[model_name]]
        test_result <- analysis_results$test_results[[model_name]]
        train_r2_value <- if (!is.null(analysis_results$train_r2[[model_name]])) round(analysis_results$train_r2[[model_name]], 2) else NA
        test_r2_value <- if (!is.null(analysis_results$test_r2[[model_name]])) round(analysis_results$test_r2[[model_name]], 2) else NA
        train_auc_value <- if (!is.null(analysis_results$train_roc[[model_name]])) round(auc(analysis_results$train_roc[[model_name]]), 2) else NA
        test_auc_value <- if (!is.null(analysis_results$test_roc[[model_name]])) round(auc(analysis_results$test_roc[[model_name]]), 2) else NA
        
        fluidRow(
          h4(model_name),
          h5("Données d'apprentissage"),
          p("R²: ", train_r2_value),
          p("AUC: ", train_auc_value),
          if (inherits(train_result, "randomForest")) {
            plotOutput(paste0("train_importance_plot_", model_name))
          } else if (inherits(train_result, "glm") || inherits(train_result, "lm")) {
            verbatimTextOutput(paste0("train_summary_", model_name))
          } else if (inherits(train_result, "nnet")) {
            verbatimTextOutput(paste0("train_summary_", model_name))
          } else if (inherits(train_result, "rpart")) {
            plotOutput(paste0("train_tree_plot_", model_name))
          },
          h5("Données de test"),
          p("R²: ", test_r2_value),
          p("AUC: ", test_auc_value),
          if (inherits(test_result, "randomForest")) {
            plotOutput(paste0("test_importance_plot_", model_name))
          } else if (inherits(test_result, "glm") || inherits(test_result, "lm")) {
            verbatimTextOutput(paste0("test_summary_", model_name))
          } else if (inherits(test_result, "nnet")) {
            verbatimTextOutput(paste0("test_summary_", model_name))
          } else if (inherits(test_result, "rpart")) {
            plotOutput(paste0("test_tree_plot_", model_name))
          }
        )
      })
      do.call(tagList, summaries)
    })
    
    output$roc_plot <- renderPlotly({
      if (!is.null(analysis_results$train_roc) && !is.null(analysis_results$test_roc)) {
        valid_models <- names(analysis_results$train_roc)[sapply(analysis_results$train_roc, function(x) !is.null(x))]
        plots <- lapply(valid_models, function(model_name) {
          train_roc <- analysis_results$train_roc[[model_name]]
          test_roc <- analysis_results$test_roc[[model_name]]
          plot_ly(x = train_roc$specificities, y = train_roc$sensitivities, type = 'scatter', mode = 'lines', name = paste(model_name, "- Train")) %>%
            add_trace(x = test_roc$specificities, y = test_roc$sensitivities, type = 'scatter', mode = 'lines', name = paste(model_name, "- Test")) %>%
            layout(title = paste("ROC Curve -", model_name),
                   xaxis = list(title = "1 - Specificity"),
                   yaxis = list(title = "Sensitivity"))
        })
        if (length(plots) > 0) {
          subplot(plots, nrows = 1, shareX = TRUE, shareY = TRUE)
        } else {
          plot_ly() %>%
            layout(title = "No valid ROC curves to display")
        }
      }
    })
    
    lapply(names(analysis_results$train_results), function(model_name) {
      train_result <- analysis_results$train_results[[model_name]]
      test_result <- analysis_results$test_results[[model_name]]
      
      if (inherits(train_result, "randomForest")) {
        output[[paste0("train_importance_plot_", model_name)]] <- renderPlot({
          varImpPlot(train_result)
        })
        output[[paste0("test_importance_plot_", model_name)]] <- renderPlot({
          varImpPlot(test_result)
        })
      } else if (inherits(train_result, "glm") || inherits(train_result, "lm")) {
        output[[paste0("train_summary_", model_name)]] <- renderPrint({
          summary(train_result)
        })
        output[[paste0("test_summary_", model_name)]] <- renderPrint({
          summary(test_result)
        })
      } else if (inherits(train_result, "nnet")) {
        output[[paste0("train_summary_", model_name)]] <- renderPrint({
          summary(train_result)
        })
        output[[paste0("test_summary_", model_name)]] <- renderPrint({
          summary(test_result)
        })
      } else if (inherits(train_result, "rpart")) {
        output[[paste0("train_tree_plot_", model_name)]] <- renderPlot({
          rpart.plot(train_result)
        })
        output[[paste0("test_tree_plot_", model_name)]] <- renderPlot({
          rpart.plot(test_result)
        })
      }
    })
  })
  
  output$download_results <- downloadHandler(
    filename = function() {
      paste("results", Sys.Date(), switch(input$result_format, "PDF" = ".pdf", "Word" = ".docx"), sep = "")
    },
    content = function(file) {
      tempReport <- file.path(tempdir(), "report_template.Rmd")
      file.copy("report_template.Rmd", tempReport, overwrite = TRUE)
      
      params <- list(
        train_results = analysis_results$train_results,
        test_results = analysis_results$test_results,
        train_r2_values = analysis_results$train_r2,
        test_r2_values = analysis_results$test_r2,
        train_auc_values = lapply(analysis_results$train_roc, auc),
        test_auc_values = lapply(analysis_results$test_roc, auc),
        train_roc_plots = analysis_results$train_roc,
        test_roc_plots = analysis_results$test_roc,
        model_summaries = lapply(analysis_results$train_results, summary)
      )
      
      if (input$result_format == "PDF") {
        rmarkdown::render(tempReport, output_file = file, params = params, envir = new.env(parent = globalenv()), output_format = "pdf_document")
      } else if (input$result_format == "Word") {
        rmarkdown::render(tempReport, output_file = file, params = params, envir = new.env(parent = globalenv()), output_format = "word_document")
      } else if (input$result_format == "Excel") {
        write.xlsx(params$results, file)
      }
    }
  )
  
  observeEvent(input$send_email, {
    req(input$email)
    tempReport <- file.path(tempdir(), "report_template.Rmd")
    file.copy("report_template.Rmd", tempReport, overwrite = TRUE)
    
    params <- list(
      train_results = analysis_results$train_results,
      test_results = analysis_results$test_results,
      train_r2_values = analysis_results$train_r2,
      test_r2_values = analysis_results$test_r2,
      train_auc_values = lapply(analysis_results$train_roc, auc),
      test_auc_values = lapply(analysis_results$test_roc, auc),
      train_roc_plots = analysis_results$train_roc,
      test_roc_plots = analysis_results$test_roc,
      model_summaries = lapply(analysis_results$train_results, summary)
    )
    
    temp_file <- tempfile(fileext = switch(input$result_format, "PDF" = ".pdf", "Word" = ".docx"))
    
    if (input$result_format == "PDF") {
      rmarkdown::render(tempReport, output_file = temp_file, params = params, envir = new.env(parent = globalenv()), output_format = "pdf_document")
    } else if (input$result_format == "Word") {
      rmarkdown::render(tempReport, output_file = temp_file, params = params, envir = new.env(parent = globalenv()), output_format = "word_document")
    } else if (input$result_format == "Excel") {
      write.xlsx(params$results, temp_file)
    }
    
    email <- compose_email(
      body = md("Voici les résultats de votre analyse."),
      footer = md("Merci d'utiliser notre interface d'analyse.")
    ) %>%
      add_attachment(temp_file)
    
    smtp_send(email,
              from = "your_email@example.com",
              to = input$email,
              subject = "Résultats de l'analyse",
              credentials = creds_key("my_smtp_key"))
  })
  
  observeEvent(input$user_guide_btn, {
    showModal(modalDialog(
      title = "Guide Utilisateur",
      tagList(
        p("Guide Utilisateur pour l'Interface Shiny d'Analyse :"),
        p("L'interface Shiny d'Analyse est conçue pour permettre aux utilisateurs de réaliser diverses analyses de classification et de régression sur des jeux de données CSV. L'application supporte plusieurs algorithmes de machine learning, dont la régression logistique, les réseaux de neurones, les arbres de décision, les random forests et la régression linéaire. L'interface est interactive et permet d'ajuster les paramètres des modèles pour affiner les analyses."),
        h4("Description des Algorithmes"),
        h5("Régression Logistique"),
        p("La régression logistique est un algorithme de classification utilisé pour prédire la probabilité qu'une observation appartienne à l'une des deux classes. Il utilise une fonction logistique pour modéliser la probabilité et est utile pour les problèmes de classification binaire."),
        p("Paramètres ajustables :"),
        p("• Seuil : Définit la probabilité seuil au-delà de laquelle une observation est classée dans la classe positive."),
        p("• Type de seuil : Permet de choisir entre alpha et 1-alpha."),
        h5("Réseaux de Neurones"),
        p("Les réseaux de neurones sont des modèles de machine learning inspirés du fonctionnement du cerveau humain. Ils sont constitués de couches de neurones interconnectés et sont utilisés pour les tâches de classification et de régression."),
        p("Paramètres ajustables :"),
        p("• Nombre de couches"),
        p("• Nombre de neurones par couche"),
        p("• Fonction d'activation (relu, sigmoid, tanh)"),
        h5("Arbres de Décision"),
        p("Les arbres de décision sont des modèles de machine learning utilisés pour la classification et la régression. Ils segmentent l'espace de décision en utilisant des règles basées sur les caractéristiques des données."),
        p("Paramètres ajustables :"),
        p("• Seuil : Définit la probabilité seuil pour la classification."),
        p("• Type de seuil : Permet de choisir entre alpha et 1-alpha."),
        h5("Random Forest"),
        p("Les random forests sont des ensembles d'arbres de décision utilisés pour améliorer la précision et réduire le sur-apprentissage. Ils utilisent un échantillonnage aléatoire des données et des caractéristiques pour construire chaque arbre."),
        p("Paramètres ajustables :"),
        p("• Nombre d'arbres"),
        h5("Régression Linéaire"),
        p("La régression linéaire est un modèle statistique utilisé pour prédire la valeur d'une variable dépendante en fonction des valeurs d'une ou plusieurs variables indépendantes. Elle est utile pour les tâches de régression."),
        p("Paramètres ajustables :"),
        p("• Nombre de cibles"),
        h4("Fonctionnalités de l'Interface"),
        h5("Accueil"),
        p("La section Accueil fournit une introduction à l'interface et explique son objectif. Elle conseille également les nouveaux utilisateurs de lire la section Aide pour mieux comprendre les conditions d'utilisation des différents modèles disponibles."),
        h5("Upload des Données"),
        p("1. Charger un Fichier CSV : Utilisez le bouton Choisir un fichier CSV pour sélectionner un fichier CSV contenant vos données."),
        p("2. Afficher les Dimensions des Données : Une fois le fichier chargé, les dimensions (nombre de lignes et de colonnes) des données sont affichées."),
        p("3. Définir le Nombre de Variables Cibles : Spécifiez le nombre de variables cibles dans les données."),
        p("4. Sélectionner le Type d'Analyse : Choisissez entre Classification et Régression."),
        h5("Ajustement des Paramètres"),
        p("Cliquez sur Ajuster les paramètres pour ouvrir une fenêtre modale où vous pouvez ajuster les paramètres des modèles sélectionnés. Les paramètres ajustables incluent les seuils, le nombre de couches et de neurones pour les réseaux de neurones, et le nombre d'arbres pour les random forests."),
        h5("Exécution de l'Analyse"),
        p("Cliquez sur RUN pour lancer l'analyse avec les paramètres et les modèles sélectionnés. Les données sont automatiquement divisées en ensembles d'entraînement et de test en fonction du pourcentage spécifié."),
        h5("Résultats"),
        p("Après l'exécution de l'analyse, les résultats sont affichés dans la section Résultats :"),
        p("• Sommaires des Modèles : Affiche un résumé des modèles entraînés."),
        p("• Courbes ROC : Affiche les courbes ROC pour évaluer les performances des modèles de classification."),
        p("• R² et AUC : Affiche les valeurs R² pour les modèles de régression et les valeurs AUC pour les modèles de classification."),
        h5("Téléchargement des Résultats"),
        p("Les utilisateurs peuvent choisir de télécharger les résultats dans les formats PDF, Word ou Excel. Cliquez sur Télécharger les résultats pour enregistrer les résultats sur votre ordinateur."),
        h5("Envoi des Résultats par Email"),
        p("Pour envoyer les résultats par email :"),
        p("1. Entrer l'Adresse Email : Saisissez l'adresse email dans le champ Email pour envoyer les résultats (optionnel)."),
        p("2. Cliquer sur Envoyer par email : Les résultats seront envoyés à l'adresse spécifiée."),
        h5("Aide"),
        p("La section Aide fournit des informations détaillées sur les formats de données acceptés, les conditions à respecter pour les données (comme l'absence de valeurs manquantes), et des conseils sur la préparation des données avant utilisation."),
        br(),
        p("Ce guide vous aide à naviguer et à utiliser efficacement l'interface Shiny d'Analyse. En suivant les étapes décrites et en ajustant les paramètres selon vos besoins, vous pouvez tirer le meilleur parti des outils d'analyse disponibles dans cette interface.")
      ),
      easyClose = TRUE,
      footer = NULL
    ))
  })
  
  observeEvent(input$maintenance_guide_btn, {
    showModal(modalDialog(
      title = "Guide de Maintenance",
      tagList(
        p("Guide de Maintenance pour l'Interface Shiny :"),
        p("Ce guide de maintenance vise à fournir des solutions pour les problèmes potentiels que les utilisateurs peuvent rencontrer lors de l'utilisation de l'interface Shiny d'Analyse. Il inclut des étapes détaillées pour diagnostiquer et résoudre divers problèmes."),
        h4("Problèmes Liés à l'Installation et à la Configuration"),
        h5("Installation de R et RStudio"),
        p("Problème : R ou RStudio n'est pas installé correctement."),
        p("Solution :"),
        p("1. Vérifiez l'installation de R :"),
        p("  • Assurez-vous que R est installé à partir du site officiel de R."),
        p("  • Essayez d'ouvrir R et exécutez version pour vérifier que R est correctement installé."),
        p("2. Vérifiez l'installation de RStudio :"),
        p("  • Assurez-vous que RStudio est installé à partir du site officiel de RStudio."),
        p("  • Essayez d'ouvrir RStudio et assurez-vous qu'il se lance correctement."),
        h5("Installation des Packages R"),
        p("Problème : Les packages R nécessaires ne sont pas installés ou ne se chargent pas correctement."),
        p("Solution :"),
        p("1. Vérifiez que tous les packages sont installés :"),
        p("  • Ouvrez RStudio et exécutez les commandes suivantes pour installer tous les packages nécessaires :"),
        p("  install.packages(c('shiny', 'shinydashboard', 'FactoMineR', 'e1071', 'randomForest',"),
        p("  'nnet', 'MASS', 'caTools', 'ggplot2', 'caret', 'pROC',"),
        p("  'rpart', 'rpart.plot', 'shinyWidgets', 'plotly',"),
        p("  'shinythemes', 'promises', 'future', 'DT',"),
        p("  'rmarkdown', 'openxlsx', 'blastula', 'mockery'))"),
        p("2. Chargez les packages :"),
        p("  • Dans RStudio, exécutez les commandes suivantes pour charger les packages :"),
        p("  library(shiny)"),
        p("  library(shinydashboard)"),
        p("  library(FactoMineR)"),
        p("  library(e1071)"),
        p("  library(randomForest)"),
        p("  library(nnet)"),
        p("  library(MASS)"),
        p("  library(caTools)"),
        p("  library(ggplot2)"),
        p("  library(caret)"),
        p("  library(pROC)"),
        p("  library(rpart)"),
        p("  library(rpart.plot)"),
        p("  library(shinyWidgets)"),
        p("  library(plotly)"),
        p("  library(shinythemes)"),
        p("  library(promises)"),
        p("  library(future)"),
        p("  library(DT)"),
        p("  library(rmarkdown)"),
        p("  library(openxlsx)"),
        p("  library(blastula)"),
        p("  library(mockery)"),
        h5("Configuration de SMTP pour l'Envoi d'Emails"),
        p("Problème : L'envoi d'emails échoue en raison de configurations incorrectes."),
        p("Solution :"),
        p("1. Activer l'accès pour les applications moins sécurisées sur Gmail :"),
        p("  • Connectez-vous à votre compte Gmail."),
        p("  • Allez à My Account Google et activez l'accès pour les applications moins sécurisées."),
        p("2. Générer un mot de passe d'application (si la vérification en deux étapes est activée) :"),
        p("  • Allez à My Account Google."),
        p("  • Suivez les instructions pour générer un mot de passe d'application."),
        p("3. Créer des clés d'authentification pour SMTP :"),
        p("  • Ouvrez RStudio et exécutez les commandes suivantes pour créer les clés d'authentification SMTP :"),
        p("  library(blastula)"),
        p("  create_smtp_creds_key("),
        p("    id = 'my_smtp_key',"),
        p("    user = 'votre_adresse_email@gmail.com',"),
        p("    provider = 'gmail',"),
        p("    use_ssl = TRUE"),
        p("  )"),
        p("4. Tester la connexion SMTP :"),
        p("  • Ouvrez l'invite de commande (CMD) et tapez les commandes suivantes :"),
        p("  openssl s_client -connect smtp.gmail.com:465"),
        p("  openssl s_client -connect smtp.gmail.com:587 -starttls smtp"),
        h4("Problèmes Liés aux Données"),
        h5("Format de Fichier CSV"),
        p("Problème : Le fichier CSV n'est pas au bon format."),
        p("Solution :"),
        p("1. Vérifiez le format du fichier CSV :"),
        p("  • Assurez-vous que le fichier est bien en format CSV avec des séparateurs ; et des décimales .."),
        p("  • Utilisez un éditeur de texte pour vérifier la structure du fichier CSV."),
        p("2. Reformatez le fichier si nécessaire :"),
        p("  • Si le fichier n'est pas correctement formaté, utilisez un outil comme Excel pour le reformater en CSV avec les bons séparateurs."),
        h5("Valeurs Manquantes"),
        p("Problème : Le fichier CSV contient des valeurs manquantes."),
        p("Solution :"),
        p("1. Identifiez les valeurs manquantes :"),
        p("  • Utilisez R pour charger le fichier CSV et vérifier les valeurs manquantes :"),
        p("  df <- read.csv('path/to/your/file.csv', sep = ';', dec = '.')"),
        p("  anyNA(df)"),
        p("2. Complétez les valeurs manquantes :"),
        p("  • Utilisez des techniques d'imputation pour compléter les valeurs manquantes ou retirez les lignes/colonnes contenant des valeurs manquantes."),
        h4("Problèmes Liés à l'Exécution de l'Analyse"),
        h5("Erreur de Chargement des Données"),
        p("Problème : Les données ne se chargent pas correctement dans l'interface."),
        p("Solution :"),
        p("1. Vérifiez que les données sont bien chargées :"),
        p("  • Assurez-vous que le chemin d'accès au fichier CSV est correct et que le fichier est accessible."),
        p("2. Affichez les dimensions des données :"),
        p("  • Vérifiez que les dimensions des données sont affichées correctement dans l'interface après le chargement."),
        h5("Erreur lors de l'Exécution de l'Analyse"),
        p("Problème : Une erreur se produit lors de l'exécution de l'analyse."),
        p("Solution :"),
        p("1. Vérifiez les paramètres d'entrée :"),
        p("  • Assurez-vous que tous les paramètres nécessaires sont correctement définis avant de lancer l'analyse."),
        p("2. Consultez les messages d'erreur :"),
        p("  • Examinez les messages d'erreur dans la console RStudio pour identifier la source du problème."),
        p("3. Ajustez les paramètres :"),
        p("  • Si une erreur se produit en raison de paramètres incorrects, ajustez les paramètres et réessayez."),
        h4("Problèmes Liés à l'Envoi d'Emails"),
        h5("Problèmes de Connexion SMTP"),
        p("Problème : La connexion au serveur SMTP échoue."),
        p("Solution :"),
        p("1. Vérifiez les paramètres SMTP :"),
        p("  • Assurez-vous que les paramètres SMTP sont corrects et que le mot de passe d'application est utilisé si nécessaire."),
        p("2. Testez la connexion SMTP :"),
        p("  • Utilisez OpenSSL pour tester la connexion au serveur SMTP (voir section précédente)."),
        h5("Échec de l'Envoi de l'Email"),
        p("Problème : L'envoi de l'email échoue."),
        p("Solution :"),
        p("1. Consultez les messages d'erreur :"),
        p("  • Examinez les messages d'erreur pour déterminer la cause de l'échec."),
        p("2. Vérifiez les adresses email :"),
        p("  • Assurez-vous que les adresses email du destinataire et de l'expéditeur sont correctes."),
        p("3. Réessayez avec des paramètres différents :"),
        p("  • Si l'envoi échoue en utilisant le port 465, essayez avec le port 587 et vice versa."),
        h4("Problèmes Liés aux Téléchargements"),
        h5("Erreur de Téléchargement des Résultats"),
        p("Problème : Une erreur se produit lors du téléchargement des résultats."),
        p("Solution :"),
        p("1. Vérifiez le format de téléchargement :"),
        p("  • Assurez-vous que le format de téléchargement sélectionné (PDF, Word, Excel) est supporté et que le fichier report_template.Rmd est accessible."),
        p("2. Consultez les messages d'erreur :"),
        p("  • Examinez les messages d'erreur pour identifier la source du problème."),
        h4("Problèmes Divers"),
        h5("Problèmes de Performance"),
        p("Problème : L'interface est lente ou ne répond pas."),
        p("Solution :"),
        p("1. Vérifiez les ressources système :"),
        p("  • Assurez-vous que le système dispose de suffisamment de mémoire et de puissance de traitement pour exécuter les analyses."),
        p("2. Optimisez les données :"),
        p("  • Réduisez la taille des données en supprimant les colonnes non nécessaires ou en utilisant des échantillons de données plus petits."),
        h5("Problèmes d'Affichage"),
        p("Problème : Les éléments de l'interface ne s'affichent pas correctement."),
        p("Solution :"),
        p("1. Vérifiez les dépendances :"),
        p("  • Assurez-vous que tous les packages et fichiers CSS/JavaScript nécessaires sont correctement chargés."),
        p("2. Consultez la console du navigateur :"),
        p("  • Examinez la console du navigateur pour détecter les erreurs de chargement des ressources."),
        br(),
        p("NB : Si après tout cela votre problème persiste, veuillez nous contacter."),
        p("e-mail : bendabire107@gmail.com")
      ),
      easyClose = TRUE,
      footer = NULL
    ))
  })
}

shinyApp(ui, server)