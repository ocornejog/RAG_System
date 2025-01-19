# generate_documents.py

import os
from pathlib import Path
import time

documents = {
    "innovation_tech.txt": """
L'Intelligence Artificielle en 2024 : Une Révolution en Marche

L'intelligence artificielle continue de transformer notre société de manière fondamentale. Les modèles de langage comme GPT-4 et Gemini représentent une avancée majeure dans la compréhension et la génération du langage naturel. Cette évolution rapide ouvre de nouvelles perspectives dans de nombreux domaines, redéfinissant notre façon de travailler et d'interagir avec la technologie.

Dans le secteur de la santé, l'IA révolutionne le diagnostic médical et accélère la recherche pharmaceutique. Les algorithmes d'apprentissage profond analysent les images médicales avec une précision remarquable, permettant une détection précoce des maladies. La recherche de nouveaux médicaments bénéficie également de ces avancées, réduisant considérablement le temps nécessaire au développement de nouveaux traitements.

Le domaine de l'éducation connaît aussi une transformation significative grâce à l'IA. L'apprentissage personnalisé devient une réalité, avec des systèmes capables d'adapter le contenu pédagogique au rythme et au style d'apprentissage de chaque étudiant. Les tuteurs virtuels offrent un soutien constant, permettant aux apprenants de progresser à leur propre rythme.

L'industrie 4.0 s'appuie fortement sur l'intelligence artificielle pour optimiser ses processus. La maintenance prédictive, rendue possible par l'analyse des données en temps réel, permet d'anticiper les pannes et de réduire les temps d'arrêt. L'automatisation intelligente augmente la productivité tout en améliorant la sécurité des travailleurs.

Cependant, cette révolution technologique soulève des questions importantes. La protection de la vie privée, l'éthique de l'IA, et son impact sur l'emploi sont des préoccupations majeures. La consommation énergétique croissante des centres de données pose également des défis environnementaux qu'il faut adresser.

L'avenir de l'IA apparaît prometteur, mais nécessite une approche équilibrée. Il est essentiel de développer ces technologies de manière responsable, en prenant en compte leurs implications sociales et environnementales. La formation continue et l'adaptation des compétences seront cruciales pour préparer la société aux changements à venir.
""",

    "environnement.txt": """
Changement Climatique : L'Urgence d'Agir

Le réchauffement climatique s'accélère à un rythme alarmant, avec des conséquences de plus en plus visibles sur notre planète. Les scientifiques observent une augmentation significative des événements météorologiques extrêmes, témoignant de l'urgence de la situation. Ces changements affectent déjà la vie quotidienne de millions de personnes à travers le monde.

La montée du niveau des océans représente une menace majeure pour les zones côtières et les îles. Les projections actuelles indiquent que de nombreuses régions pourraient devenir inhabitables dans les décennies à venir. Cette situation critique s'accompagne d'une érosion accélérée des côtes et d'une augmentation des inondations dans les zones littorales.

La perte de biodiversité atteint des niveaux sans précédent. De nombreuses espèces animales et végétales disparaissent à un rythme cent fois supérieur au taux naturel d'extinction. Cet appauvrissement de la biodiversité menace l'équilibre des écosystèmes et, par conséquent, la sécurité alimentaire mondiale.

Face à ces défis, des solutions concrètes émergent. La transition vers les énergies renouvelables s'accélère, avec le développement de parcs éoliens et solaires toujours plus efficaces. L'adoption de véhicules électriques et le développement des transports en commun contribuent à réduire les émissions de gaz à effet de serre.

Les citoyens jouent un rôle crucial dans cette lutte contre le changement climatique. De plus en plus de personnes adoptent des modes de consommation responsables, privilégiant les produits locaux et durables. Le mouvement vers le zéro déchet prend de l'ampleur, démontrant une prise de conscience collective.

L'action politique et la coopération internationale sont essentielles pour faire face à ce défi global. Les accords climatiques, bien qu'imparfaits, constituent une base pour une action coordonnée. L'engagement des jeunes générations dans ce combat donne espoir pour l'avenir.
""",

    "culture_francaise.txt": """
La Culture Française Contemporaine

La France maintient sa position unique dans le paysage culturel mondial, alliant tradition et modernité avec une élégance caractéristique. La culture française contemporaine continue d'évoluer et de se réinventer, tout en préservant son riche héritage historique.

La littérature française connaît un renouveau remarquable. Une nouvelle génération d'auteurs émerge, proposant des œuvres qui explorent les enjeux contemporains tout en maintenant l'excellence stylistique française. Les maisons d'édition indépendantes jouent un rôle crucial dans la découverte de nouvelles voix, enrichissant le paysage littéraire.

Le cinéma français reste une référence mondiale. Les réalisateurs contemporains osent explorer de nouveaux territoires narratifs et visuels, tout en conservant cette touche distinctive du cinéma français. Les coproductions internationales permettent une diffusion plus large de ces œuvres, contribuant au rayonnement culturel de la France.

Les arts vivants démontrent une vitalité exceptionnelle. Le Festival d'Avignon et le Festival de Cannes continuent d'attirer les talents du monde entier. La scène musicale française se diversifie, mêlant genres traditionnels et nouvelles tendances. Le théâtre contemporain repousse les limites de la création, proposant des expériences innovantes.

La gastronomie française, inscrite au patrimoine de l'UNESCO, poursuit son évolution. Les chefs contemporains réinterprètent les classiques tout en intégrant des préoccupations modernes comme la durabilité et l'éthique alimentaire. Cette fusion entre tradition culinaire et innovation crée une cuisine vivante et en constante évolution.

L'art de vivre à la française reste une source d'inspiration mondiale. Les nouveaux créateurs de mode, designers et artistes contribuent à maintenir Paris comme capitale culturelle internationale, tout en apportant leur vision contemporaine à cet héritage séculaire.
""",

    "sante_publique.txt": """
Les Défis de la Santé Publique Moderne

Le système de santé français fait face à des défis complexes et multiformes dans un contexte de transformation rapide de la société. L'accès aux soins reste une préoccupation majeure, particulièrement dans les zones rurales où la désertification médicale s'accentue.

La question des déserts médicaux devient cruciale. De nombreuses régions souffrent d'un manque critique de professionnels de santé, créant des inégalités territoriales importantes. Les délais d'attente pour consulter certains spécialistes s'allongent, tandis que le coût des traitements innovants pose la question de l'équité dans l'accès aux soins.

La prévention prend une place de plus en plus importante dans les politiques de santé publique. Les campagnes de sensibilisation se multiplient pour lutter contre la sédentarité, le tabagisme et les mauvaises habitudes alimentaires. Les programmes de dépistage précoce des maladies graves se développent, permettant une meilleure prise en charge.

La santé mentale émerge comme une priorité majeure. La reconnaissance des troubles psychologiques s'améliore, même si des progrès restent à faire. Les thérapies alternatives gagnent en reconnaissance, offrant de nouvelles options aux patients. Le soutien aux aidants familiaux devient également un enjeu crucial dans une société vieillissante.

L'innovation médicale transforme profondément les pratiques. La télémédecine se développe rapidement, offrant de nouvelles solutions pour l'accès aux soins. Les thérapies géniques ouvrent des perspectives prometteuses pour le traitement de maladies jusqu'ici incurables. L'intelligence artificielle commence à être utilisée pour le diagnostic et la recherche médicale.

L'avenir de la santé publique repose sur un équilibre délicat entre l'innovation technologique et l'approche humaine des soins. La formation des professionnels de santé évolue pour intégrer ces nouvelles dimensions, tout en préservant la relation patient-soignant au cœur du système de santé.
""",

    "economie_numerique.txt": """
L'Économie Numérique en Transformation

L'économie numérique bouleverse profondément les modèles traditionnels de production et de consommation. Cette transformation digitale touche tous les secteurs d'activité, créant de nouvelles opportunités mais aussi de nouveaux défis pour les entreprises et les consommateurs.

Le commerce en ligne connaît une croissance exponentielle. Les habitudes d'achat évoluent rapidement, poussant les entreprises à repenser leur présence digitale. Les nouveaux modes de paiement se multiplient, offrant toujours plus de flexibilité aux consommateurs. La personnalisation de l'expérience client devient un enjeu majeur pour se démarquer dans un marché hautement concurrentiel.

Le travail à distance s'est imposé comme une nouvelle norme dans de nombreux secteurs. Les outils collaboratifs ne cessent de s'améliorer, permettant une coordination efficace des équipes dispersées géographiquement. Cette évolution soulève des questions importantes sur l'équilibre entre vie professionnelle et personnelle, ainsi que sur l'organisation du travail.

L'écosystème des startups français montre un dynamisme remarquable. Les initiatives de financement participatif se multiplient, offrant de nouvelles opportunités aux entrepreneurs. Les incubateurs et accélérateurs jouent un rôle crucial dans l'accompagnement des jeunes entreprises innovantes, contribuant à la vitalité de l'économie numérique.

La cybersécurité devient un enjeu critique face à la multiplication des menaces en ligne. La protection des données personnelles préoccupe autant les entreprises que les particuliers. La fracture numérique reste un défi majeur, nécessitant des politiques publiques adaptées pour garantir l'accès de tous aux opportunités du numérique.

L'avenir de l'économie numérique repose sur la capacité des organisations à s'adapter continuellement aux innovations technologiques. La formation continue devient essentielle pour maintenir les compétences à jour dans un environnement en constante évolution.
""",

    "tweets.txt": """
Je suis tellement content de mon nouveau téléphone ! #tech
Le match de ce soir était incroyable ! #football
J’adore cette nouvelle série sur Netflix. #divertissement
Il pleut encore aujourd'hui, quel temps maussade. #météo
L'intelligence artificielle va changer le monde. #IA
Le réchauffement climatique est un problème sérieux. #environnement
Je viens de finir de lire un livre fascinant sur l'économie. #lecture
La conférence sur la tech d’aujourd’hui était inspirante ! #tech
Le dernier film de science-fiction que j'ai vu était génial ! #cinéma
Le projet sur lequel je travaille avance bien. #travail
""",

    "lois.txt": """
Article 1: Nul ne peut être condamné pour une action ou une omission qui, au moment où elle a été commise, ne
constituait pas une infraction pénale selon le droit national ou international.
Article 2: Toute personne accusée d'une infraction pénale est présumée innocente jusqu'à ce que sa culpabilité ait
été légalement établie.
Article 3: Toute personne privée de liberté doit être traitée avec humanité et avec le respect de la dignité
inhérente à la personne humaine.
Article 4: Aucune peine privative de liberté ne peut être infligée pour inexécution d'une obligation contractuelle.
Article 5: Toute personne a droit à un recours effectif devant une instance nationale pour les actes violant les
droits fondamentaux.
Article 6: Le droit à la vie est protégé par la loi. Nul ne peut être intentionnellement privé de la vie, sauf en
exécution d'une condamnation à mort.
Article 7: La torture et les peines ou traitements inhumains ou dégradants sont interdits en toutes circonstances.
Article 8: Toute personne a droit au respect de sa vie privée et familiale, de son domicile et de sa correspondance.
Article 9: Toute personne a droit à la liberté d'expression, sous réserve de ne pas porter atteinte à l’ordre
public.
Article 10: La liberté de pensée, de conscience et de religion est protégée par la loi.
""",

    "donnees_melangees.txt": """
Je suis tellement content de mon nouveau téléphone ! #tech
Article 1: Nul ne peut être condamné pour une action ou une omission qui, au moment où elle a été commise, ne
constituait pas une infraction pénale.
Le match de ce soir était incroyable ! #football
Article 3: Toute personne privée de liberté doit être traitée avec humanité et dignité.
L'intelligence artificielle va changer le monde. #IA
Le réchauffement climatique est un problème sérieux. #environnement
Article 5: Toute personne a droit à un recours effectif devant une instance nationale pour les actes violant les
droits fondamentaux.
La conférence sur la tech d’aujourd’hui était inspirante ! #tech
Le dernier film de science-fiction que j'ai vu était génial ! #cinéma
"""
}

def create_documents():
    # Create documents directory if it doesn't exist
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Write each document to a file with delay
    for filename, content in documents.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"Created: {filename}")
        
        # Attendre 2 secondes entre chaque fichier
        time.sleep(2)  # Délai de 2 secondes
        print(f"Waiting 2 seconds before next file...")

if __name__ == "__main__":
    create_documents()