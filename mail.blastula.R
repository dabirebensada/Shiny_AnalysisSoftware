library(blastula)

# Configurer l'envoi d'emails - exécuter dans R (pas dans Shiny)
# IMPORTANT pour Gmail : vous DEVEZ utiliser un "Mot de passe d'application"
# (pas votre mot de passe Gmail habituel) :
# 1. Allez sur https://myaccount.google.com/security
# 2. Activez la validation en 2 étapes si nécessaire
# 3. Mots de passe d'application > Générer un mot de passe
# 4. Copiez les 16 caractères et collez-les quand R vous le demande

create_smtp_creds_key(
  id = "my_smtp_key",
  user = "bendabire107@gmail.com",
  provider = "gmail",
  use_ssl = TRUE,
  overwrite = TRUE
)

# Vérifier que la clé a été créée
view_credential_keys()
