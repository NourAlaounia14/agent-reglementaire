# Guide RGPD pour les Startups - CNIL (Commission Nationale de l'Informatique et des Libertés)

## 1. Principes fondamentaux pour le traitement des données
Toute startup traitant des données personnelles doit respecter le Règlement Général sur la Protection des Données (RGPD). 
- **Minimisation des données :** Ne collectez que les données strictement nécessaires au service.
- **Transparence :** Informez clairement les utilisateurs via une politique de confidentialité accessible.
- **Sécurité :** Mettez en place des mesures techniques (chiffrement des données en transit et au repos, hachage des mots de passe).

## 2. Cas particulier des données de santé (HealthTech / Dispositifs Médicaux)
Si votre application traite des données de santé (comme des données cardiaques, électrocardiogrammes, rythme cardiaque) :
- Ce sont des **données sensibles** au sens de l'article 9 du RGPD.
- Leur traitement est **par principe interdit**, sauf exceptions : recueil du **consentement explicite** de la personne concernée, ou nécessité pour des motifs d'intérêt public dans le domaine de la santé publique.
- **Hébergement agréé :** En France, l'hébergement de données de santé (HDS) par un tiers impose de confier ces données à un hébergeur certifié HDS.
- **DPO (Délégué à la Protection des Données) :** La désignation d'un DPO est fortement recommandée, voire obligatoire si l'activité de base consiste au traitement à grande échelle de données sensibles.