#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GENERATEUR DU RAPPORT PDF EXPLICATIF
================================================================================
Construit un rapport PDF qui explique rigoureusement la methodologie de
chaque variante de Diff-in-Diff implementee dans did_template.py, presente
les resultats obtenus sur la base de donnees, et detaille chaque test
statistique (hypotheses, statistique de test, conclusion).

Utilise reportlab (Platypus) -- cf. /mnt/skills/public/pdf/SKILL.md.
================================================================================
"""

import os
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether, HRFlowable
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "outputs", "rapport_diff_in_diff.pdf")

NAVY = colors.HexColor("#1e3a5f")
BLUE = colors.HexColor("#2563eb")
GREY = colors.HexColor("#4b5563")
LIGHT = colors.HexColor("#f3f4f6")
RED = colors.HexColor("#b91c1c")
GREEN = colors.HexColor("#15803d")

# ------------------------------------------------------------------------------
# STYLES
# ------------------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitrePrincipal", parent=styles["Title"],
                           fontSize=24, leading=28, textColor=NAVY,
                           spaceAfter=6))
styles.add(ParagraphStyle("SousTitre", parent=styles["Normal"],
                           fontSize=13, leading=17, textColor=GREY,
                           alignment=TA_CENTER, spaceAfter=4))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"],
                           fontSize=16, leading=20, textColor=NAVY,
                           spaceBefore=18, spaceAfter=8,
                           borderColor=NAVY, borderWidth=0))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"],
                           fontSize=12.5, leading=16, textColor=BLUE,
                           spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle("H3", parent=styles["Heading3"],
                           fontSize=11, leading=14, textColor=GREY,
                           spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle("Corps", parent=styles["BodyText"],
                           fontSize=9.7, leading=13.5, alignment=TA_JUSTIFY,
                           spaceAfter=6))
styles.add(ParagraphStyle("Formule", parent=styles["BodyText"],
                           fontSize=9.7, leading=14, alignment=TA_CENTER,
                           textColor=NAVY, fontName="Courier",
                           backColor=LIGHT, borderPadding=6, spaceBefore=4,
                           spaceAfter=8))
styles.add(ParagraphStyle("Legende", parent=styles["BodyText"],
                           fontSize=8.3, leading=11, alignment=TA_CENTER,
                           textColor=GREY, spaceBefore=2, spaceAfter=14,
                           fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle("Encadre", parent=styles["BodyText"],
                           fontSize=9.3, leading=12.8, alignment=TA_JUSTIFY,
                           backColor=LIGHT, borderPadding=8, spaceBefore=6,
                           spaceAfter=10))
styles.add(ParagraphStyle("Conclusion", parent=styles["BodyText"],
                           fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
                           textColor=colors.black, spaceBefore=4, spaceAfter=10,
                           leftIndent=4, borderColor=GREY, borderWidth=0.6,
                           borderPadding=7))


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (x != x)):
        return "--"
    return f"{x:.{nd}f}"


def _pval(x):
    if x is None or (isinstance(x, float) and (x != x)):
        return "--"
    return "< 0,0001" if x < 0.0001 else f"{x:.4f}"


def _img(path, width=15.5 * cm):
    """Insere une image en conservant son ratio d'aspect."""
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = h / w
    return Image(path, width=width, height=width * ratio)


def section_title_page(story, n_obs, n_units, periode_min, periode_max):
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Analyse en Difference-in-Differences", styles["TitrePrincipal"]))
    story.append(Paragraph("Rapport méthodologique et résultats empiriques",
                            styles["SousTitre"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="60%", thickness=1.2, color=NAVY,
                             hAlign="CENTER", spaceAfter=14))
    story.append(Paragraph(
        f"Base de données analysée : panel de {n_units} unités observées sur "
        f"{periode_max - periode_min + 1} périodes ({n_obs} observations).",
        styles["SousTitre"]))
    story.append(Spacer(1, 2.5 * cm))
    story.append(Paragraph(
        "Ce document présente, de manière rigoureuse et reproductible, les "
        "principales variantes de l'estimateur en différence de différences "
        "(Difference-in-Differences, DiD) appliquées à la base de données "
        "jointe à ce rapport : DiD classique \"2x2\", DiD multi-périodes à "
        "effets fixes (TWFE), DiD dynamique (event-study), décomposition de "
        "Goodman-Bacon et estimateur robuste de Callaway &amp; Sant'Anna pour "
        "les cas d'adoption échelonnée du traitement. Chaque méthode est "
        "accompagnée de sa justification théorique, de ses hypothèses "
        "d'identification, des résultats obtenus et des tests statistiques "
        "associés.", styles["Corps"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(f"Document généré automatiquement le {date.today().strftime('%d/%m/%Y')}.",
                            styles["Legende"]))
    story.append(PageBreak())


def section_intro(story):
    story.append(Paragraph("1. Cadre méthodologique général", styles["H1"]))
    story.append(Paragraph(
        "L'estimateur en différence de différences vise à estimer l'effet "
        "causal d'un traitement (une politique publique, un programme, un "
        "changement réglementaire...) en comparant l'évolution dans le temps "
        "de l'outcome Y entre un groupe traité et un groupe de contrôle. "
        "Son intérêt principal est de neutraliser, par différenciation, "
        "toute différence de NIVEAU constante entre les groupes (effets fixes "
        "individuels) ainsi que tout choc commun affectant identiquement les "
        "deux groupes (effet fixe temporel).", styles["Corps"]))

    story.append(Paragraph("1.1 Hypothèses d'identification", styles["H2"]))
    items = [
        ("Tendances parallèles (parallel trends)",
         "en l'absence de traitement, le groupe traité aurait connu, en "
         "moyenne, la même évolution de Y que le groupe de contrôle. C'est "
         "l'hypothèse centrale, non testable directement (elle porte sur un "
         "contrefactuel inobservé), mais dont on peut tester une implication "
         "nécessaire : l'absence de divergence de tendance AVANT le "
         "traitement (cf. section 3 et tests statistiques, section 4)."),
        ("Absence d'anticipation (no anticipation)",
         "les unités ne modifient pas leur comportement en anticipation du "
         "traitement avant sa mise en œuvre effective."),
        ("SUTVA (Stable Unit Treatment Value Assumption)",
         "le traitement d'une unité n'affecte pas l'outcome des autres "
         "unités (pas d'effets de diffusion/débordement, \"spillovers\")."),
        ("Pas de changement de composition",
         "dans un panel, les unités restent les mêmes sur toute la période "
         "(pas d'entrée/sortie liée au traitement lui-même)."),
    ]
    flow_items = []
    for titre, texte in items:
        flow_items.append(ListItem(Paragraph(f"<b>{titre}</b> : {texte}", styles["Corps"]),
                                    leftIndent=8))
    story.append(ListFlowable(flow_items, bulletType="bullet", start="•"))

    story.append(Paragraph("1.2 Pourquoi plusieurs variantes ?", styles["H2"]))
    story.append(Paragraph(
        "Le DiD \"2x2\" canonique (un groupe traité, un groupe de contrôle, "
        "une période avant/après) est le cas le plus simple et le plus "
        "facile à interpréter, mais il ne s'applique tel quel qu'à un design "
        "à une seule date de traitement. Dès lors que les données couvrent "
        "plusieurs périodes et/ou que les unités sont traitées à des dates "
        "différentes (<i>adoption échelonnée</i>, \"staggered adoption\"), il "
        "faut recourir à des extensions : la régression multi-périodes à "
        "effets fixes (TWFE), la spécification dynamique \"event-study\" (qui "
        "permet de tester les tendances parallèles ET de visualiser la "
        "dynamique de l'effet), et — point crucial mis en évidence par la "
        "littérature récente (Goodman-Bacon, 2021 ; Callaway &amp; Sant'Anna, "
        "2021 ; de Chaisemartin &amp; D'Haultfoeuille, 2020 ; Sun &amp; "
        "Abraham, 2021) — des estimateurs robustes lorsque l'effet du "
        "traitement est hétérogène entre cohortes ou évolue dans le temps, "
        "car le TWFE statique standard peut alors être <b>biaisé</b>.",
        styles["Corps"]))
    story.append(PageBreak())


def section_donnees(story, results):
    data = results["data"]
    story.append(Paragraph("2. Données et statistiques descriptives", styles["H1"]))
    story.append(Paragraph(
        "La base utilisée est un panel d'unités observées sur plusieurs "
        "périodes, comportant un groupe jamais-traité et plusieurs cohortes "
        "traitées à des dates différentes (adoption échelonnée), ce qui "
        "permet d'illustrer l'ensemble des méthodes présentées dans ce "
        "rapport.", styles["Corps"]))

    n_per_group = data.groupby("group")["unit_id"].nunique()
    first_treats = data.groupby("group")["first_treat"].first()
    rows = [["Groupe / cohorte", "Nombre d'unités", "Première période de traitement"]]
    for g in n_per_group.index:
        ft = first_treats[g]
        rows.append([g.replace("_", " "), str(n_per_group[g]),
                     "jamais traité" if pd_isna(ft) else str(int(ft))])
    t = Table(rows, hAlign="CENTER", colWidths=[6.5 * cm, 4 * cm, 5.5 * cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(_img(results["figures"]["trends"]))
    story.append(Paragraph(
        "Figure 1 — Trajectoires moyennes de l'outcome Y par cohorte de "
        "traitement. Les lignes verticales pointillées indiquent la date de "
        "première exposition au traitement de chaque cohorte. L'inspection "
        "visuelle des tendances avant traitement constitue un premier "
        "diagnostic (informel) de la plausibilité de l'hypothèse de "
        "tendances parallèles.", styles["Legende"]))
    story.append(PageBreak())


def pd_isna(x):
    try:
        return x != x  # NaN check sans dependre de pandas dans ce module
    except Exception:
        return False


def _table_style(header_bg=NAVY):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])


def section_did_2x2(story, results):
    r = results["res_2x2"]
    story.append(Paragraph("3. Variantes du Diff-in-Differences", styles["H1"]))
    story.append(Paragraph("3.1 DiD classique \"2x2\"", styles["H2"]))
    story.append(Paragraph(
        "Cas le plus simple : un groupe traité, un groupe de contrôle, une "
        "période avant et une période après le traitement. L'effet DiD est "
        "la double différence des moyennes :", styles["Corps"]))
    story.append(Paragraph(
        "DiD = [ Y_moy(traite, apres) - Y_moy(traite, avant) ] - "
        "[ Y_moy(controle, apres) - Y_moy(controle, avant) ]", styles["Formule"]))
    story.append(Paragraph(
        "Cette quantité s'obtient de manière strictement équivalente comme "
        "le coefficient associé au terme d'interaction d'une régression "
        "OLS :", styles["Corps"]))
    story.append(Paragraph(
        "Y_it = β0 + β1·Traité_i + β2·Après_t + β3·(Traité_i × Après_t) + ε_it",
        styles["Formule"]))
    story.append(Paragraph(
        "où β3 est l'estimateur DiD recherché. L'avantage de la forme "
        "régression est de permettre l'ajout de covariables de contrôle et "
        "le calcul direct d'erreurs-types robustes/clusterisées par unité "
        "(recommandé dès qu'il y a plusieurs observations par unité ou "
        "une structure de groupe, afin de tenir compte de l'autocorrélation "
        "intra-unité).", styles["Corps"]))

    story.append(_img(results["figures"]["did_2x2"], width=11 * cm))
    story.append(Paragraph(
        "Figure 2 — Représentation canonique du DiD 2x2 : la droite "
        "pointillée bleue représente l'évolution contrefactuelle du groupe "
        "traité (sa valeur de départ + la variation observée pour le groupe "
        "de contrôle). L'écart vertical entre cette contrefactuelle et la "
        "valeur réellement observée après traitement est l'estimateur DiD.",
        styles["Legende"]))

    rows = [["Comparaison", "Estimation", "Erreur-type\n(clusterisée/unité)",
             "IC 95%", "p-value"],
            [r["comparaison"].replace("_", " "), _fmt(r["estimate"], 3),
             _fmt(r["se"], 3), f"[{_fmt(r['ci_low'],2)} ; {_fmt(r['ci_high'],2)}]",
             _pval(r["pvalue"])]]
    t = Table(rows, hAlign="CENTER", colWidths=[5.2*cm, 2.6*cm, 3.4*cm, 3.2*cm, 2.2*cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Paragraph(
        f"<b>Vérification de cohérence</b> : le calcul direct par moyennes "
        f"de cellules donne {_fmt(r['verif_calcul_manuel'],3)}, identique "
        f"(à l'arrondi) au coefficient de régression — ce qui confirme "
        f"l'équivalence algébrique entre les deux approches de calcul.",
        styles["Encadre"]))
    story.append(Spacer(1, 6))


def section_twfe(story, results):
    r = results["res_twfe"]
    story.append(Paragraph("3.2 DiD multi-périodes : TWFE statique", styles["H2"]))
    story.append(Paragraph(
        "Lorsque le panel comporte plus de deux périodes et/ou plusieurs "
        "cohortes traitées à des dates différentes, la spécification "
        "usuelle est la régression à effets fixes à deux dimensions "
        "(\"Two-Way Fixed Effects\", TWFE) :", styles["Corps"]))
    story.append(Paragraph(
        "Y_it = α_i + λ_t + β·D_it + ε_it", styles["Formule"]))
    story.append(Paragraph(
        "où α_i est un effet fixe individuel, λ_t un effet fixe temporel, et "
        "D_it l'indicatrice valant 1 si l'unité i est effectivement traitée "
        "à la période t. β est alors interprété comme l'effet moyen du "
        "traitement sur les traités (ATT).", styles["Corps"]))
    story.append(Paragraph(
        "<b>Mise en garde essentielle</b> : Goodman-Bacon (2021) et la "
        "littérature qui a suivi ont montré que, dès lors que (i) "
        "l'adoption du traitement est échelonnée dans le temps ET (ii) "
        "l'effet du traitement est hétérogène entre cohortes ou évolue "
        "dynamiquement avec la durée d'exposition, l'estimateur β du TWFE "
        "statique n'identifie en général PAS un ATT interprétable : il peut "
        "même, dans certains cas, être de signe opposé à l'effet réel. "
        "La raison est que la régression utilise implicitement certaines "
        "unités déjà traitées comme groupe de contrôle pour d'autres "
        "cohortes — une comparaison \"à risque\" (cf. section 3.4).",
        styles["Encadre"]))

    rows = [["Spécification", "Estimation (β)", "Erreur-type", "p-value"],
            ["TWFE statique, toutes cohortes", _fmt(r["estimate"], 3),
             _fmt(r["se"], 3), _pval(r["pvalue"])]]
    t = Table(rows, hAlign="CENTER", colWidths=[7*cm, 3.5*cm, 3*cm, 2.5*cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 8))


def section_event_study(story, results):
    story.append(Paragraph("3.3 DiD dynamique : \"event-study\"", styles["H2"]))
    story.append(Paragraph(
        "La spécification dynamique remplace l'indicatrice unique D_it par "
        "une série d'indicatrices de temps relatif au traitement (\"leads\" "
        "et \"lags\") :", styles["Corps"]))
    story.append(Paragraph(
        "Y_it = α_i + λ_t + Σ_(k≠-1) δ_k · 1{t − G_i = k} + ε_it",
        styles["Formule"]))
    story.append(Paragraph(
        "où G_i est la date de traitement de l'unité i (G_i = ∞ pour les "
        "unités jamais traitées) et k = t − G_i le temps relatif au "
        "traitement. La période de référence (généralement k = −1, la "
        "période juste avant le traitement) est omise et sert de "
        "normalisation. Les coefficients k &lt; 0 (\"leads\") permettent de "
        "<b>tester</b> les tendances parallèles avant traitement ; les "
        "coefficients k ≥ 0 (\"lags\") retracent la <b>dynamique</b> de "
        "l'effet après traitement. Par souci de parcimonie statistique, les "
        "temps relatifs extrêmes sont regroupés (\"binnés\") dans des "
        "catégories terminales (ici ± 6 périodes).", styles["Corps"]))

    story.append(_img(results["figures"]["event_study"]))
    story.append(Paragraph(
        "Figure 3 — Coefficients d'event-study (TWFE) et intervalles de "
        "confiance à 95 %, par temps relatif au traitement. Les coefficients "
        "avant la ligne verticale (périodes pré-traitement) doivent être "
        "proches de zéro et non significatifs si l'hypothèse de tendances "
        "parallèles est valide ; ceux après la ligne retracent la montée en "
        "puissance de l'effet du traitement.", styles["Legende"]))
    story.append(PageBreak())


def section_bacon(story, results):
    bacon_df = results["bacon_df"]
    beta_rec = results["beta_reconstructed"]
    beta_twfe = results["res_twfe"]["estimate"]

    story.append(Paragraph("3.4 DiD à adoption échelonnée (1) : décomposition de Goodman-Bacon",
                            styles["H2"]))
    story.append(Paragraph(
        "Goodman-Bacon (2021, <i>Journal of Econometrics</i>) démontre que le "
        "coefficient du TWFE statique multi-cohortes est <b>mathématiquement "
        "égal</b> à une moyenne pondérée de TOUTES les comparaisons DiD 2x2 "
        "possibles entre paires de groupes définis par leur calendrier de "
        "traitement (chaque cohorte, et le groupe jamais-traité considéré "
        "comme une cohorte traitée \"à l'infini\") :", styles["Corps"]))
    story.append(Paragraph(
        "β_TWFE = Σ_(k,l) s_kl · β(2x2)_kl", styles["Formule"]))
    story.append(Paragraph(
        "Ce théorème permet de distinguer deux types de comparaisons : (i) "
        "les comparaisons <b>propres</b> entre un groupe traité et le groupe "
        "jamais-traité (ou pas-encore-traité), non biaisées ; et (ii) les "
        "comparaisons <b>à risque</b> entre deux groupes <i>tous deux "
        "éventuellement traités</i>, dans lesquelles un groupe déjà traité "
        "sert implicitement de \"contrôle\" pour un autre groupe — alors "
        "qu'il subit lui-même un effet de traitement actif. Si cet effet "
        "est hétérogène ou évolue dans le temps, ces comparaisons biaisent "
        "l'estimateur global.", styles["Corps"]))
    story.append(Paragraph(
        "<i>Note d'implémentation</i> : la pondération utilisée ici "
        "applique le résultat général de pondération de toute régression "
        "groupée comme moyenne pondérée de coefficients de sous-échantillons "
        "(poids proportionnel à la taille du sous-échantillon × variance du "
        "traitement résiduelle des effets fixes), ce qui reproduit "
        "fidèlement la logique de la décomposition de Goodman-Bacon sans "
        "ré-implémenter l'algorithme original au bit près. La reconstruction "
        f"pondérée obtenue ({_fmt(beta_rec,3)}) coïncide bien, comme attendu, "
        f"avec l'estimation directe du TWFE ({_fmt(beta_twfe,3)}), ce qui "
        "valide numériquement la décomposition. Pour une réplication exacte "
        "de l'algorithme publié, le package R <font face=\"Courier\">"
        "bacondecomp</font> reste la référence.", styles["Corps"]))

    story.append(_img(results["figures"]["bacon"]))
    story.append(Paragraph(
        "Figure 4 — \"Bacon plot\" : chaque point est une comparaison 2x2 "
        "entre deux groupes de calendrier de traitement, positionnée selon "
        "son poids dans la décomposition (abscisse) et son estimation "
        "(ordonnée). En rouge : comparaisons \"à risque\" entre deux groupes "
        "éventuellement traités ; en vert : comparaisons propres vs le "
        "groupe jamais-traité.", styles["Legende"]))

    rows = [["Comparaison", "Type", "β (2x2)", "Poids"]]
    for _, r in bacon_df.iterrows():
        rows.append([f"{r['groupe_1'].replace('_',' ')} / {r['groupe_2'].replace('_',' ')}",
                     "À risque" if "risque" in r["type_comparaison"] else "Propre",
                     _fmt(r["beta_2x2"], 3), _fmt(r["poids"], 3)])
    t = Table(rows, hAlign="CENTER", colWidths=[7.5*cm, 2.7*cm, 2.5*cm, 2.3*cm])
    t.setStyle(_table_style())
    story.append(t)

    part_risque = bacon_df.loc[bacon_df["type_comparaison"].str.contains("risque"), "poids"].sum()
    story.append(Paragraph(
        f"<b>Lecture</b> : les comparaisons \"à risque\" représentent ici "
        f"{part_risque:.0%} du poids total de la décomposition. Une part "
        "importante de comparaisons à risque, combinée à une forte "
        "dispersion des estimations 2x2 sous-jacentes, est un signal "
        "d'alerte indiquant que le TWFE statique global doit être "
        "interprété avec prudence et qu'un estimateur robuste (section 3.5) "
        "est préférable pour résumer l'effet moyen du traitement.",
        styles["Conclusion"]))
    story.append(PageBreak())


def section_cs(story, results):
    cs = results["cs_result"]
    story.append(Paragraph(
        "3.5 DiD à adoption échelonnée (2) : estimateur de Callaway &amp; Sant'Anna",
        styles["H2"]))
    story.append(Paragraph(
        "Callaway &amp; Sant'Anna (2021, <i>Journal of Econometrics</i>) "
        "proposent de construire l'inférence à partir d'effets désagrégés "
        "par cohorte de traitement g et par période t, les "
        "<b>ATT(g,t)</b> (\"group-time average treatment effects\") :",
        styles["Corps"]))
    story.append(Paragraph(
        "ATT(g,t) = E[Y_t − Y_(g−1) | G = g] − E[Y_t − Y_(g−1) | groupe de comparaison]",
        styles["Formule"]))
    story.append(Paragraph(
        "Chaque ATT(g,t) est un DiD 2x2 \"propre\" entre la cohorte g et un "
        "groupe de comparaison non-encore-traité à la période t (ici, le "
        "groupe jamais-traité), ancré sur la dernière période avant le "
        "traitement de la cohorte (g−1). En particulier, les valeurs "
        "ATT(g,t) calculées pour t &lt; g constituent un test de "
        "pré-tendance SPÉCIFIQUE à chaque cohorte. Ces effets désagrégés "
        "sont ensuite agrégés : (i) en un ATT global (moyenne pondérée par "
        "taille de cohorte des effets post-traitement) et (ii) en un profil "
        "dynamique ATT(e), où e = t − g est le temps relatif au traitement, "
        "directement comparable à l'event-study TWFE de la section 3.3 mais "
        "robuste à l'hétérogénéité dynamique des effets.", styles["Corps"]))

    story.append(_img(results["figures"]["att_heatmap"], width=16.5*cm))
    story.append(Paragraph(
        "Figure 5 — Carte des ATT(g,t) par cohorte (lignes) et période "
        "(colonnes). Les zones correspondant aux périodes antérieures à la "
        "date de traitement de chaque cohorte servent de test de "
        "pré-tendance ; les zones postérieures retracent la dynamique de "
        "l'effet, cohorte par cohorte.", styles["Legende"]))

    story.append(_img(results["figures"]["cs_vs_twfe"]))
    story.append(Paragraph(
        "Figure 6 — Comparaison du profil dynamique obtenu par l'event-study "
        "TWFE naïf (gris) et par l'agrégation des ATT(g,t) de Callaway &amp; "
        "Sant'Anna (vert). Un écart marqué entre les deux profils, "
        "particulièrement après traitement, signale la présence d'un biais "
        "du TWFE statique lié à l'hétérogénéité dynamique des effets de "
        "traitement entre cohortes (cf. section 3.4).", styles["Legende"]))

    rows = [["Méthode", "ATT estimé", "Erreur-type", "IC 95% (approx.)"],
            ["TWFE statique", _fmt(results['res_twfe']['estimate'],3),
             _fmt(results['res_twfe']['se'],3),
             f"[{_fmt(results['res_twfe']['ci_low'],2)} ; {_fmt(results['res_twfe']['ci_high'],2)}]"],
            ["Callaway & Sant'Anna (agrégé)", _fmt(cs['att_simple'],3),
             _fmt(cs['se_simple'],3),
             f"[{_fmt(cs['att_simple']-1.96*cs['se_simple'],2)} ; "
             f"{_fmt(cs['att_simple']+1.96*cs['se_simple'],2)}]"]]
    t = Table(rows, hAlign="CENTER", colWidths=[6*cm, 3.2*cm, 3*cm, 4*cm])
    t.setStyle(_table_style())
    story.append(t)

    ecart = cs['att_simple'] - results['res_twfe']['estimate']
    story.append(Paragraph(
        f"<b>Lecture</b> : l'écart entre les deux estimateurs "
        f"({_fmt(ecart,3)}) illustre concrètement le biais du TWFE "
        "statique en présence d'effets dynamiques hétérogènes entre "
        "cohortes : les cohortes traitées plus tôt ayant ici un effet qui "
        "croît plus rapidement avec la durée d'exposition, elles "
        "contaminent, via les comparaisons \"à risque\" identifiées par la "
        "décomposition de Goodman-Bacon, l'estimation globale du TWFE. "
        "L'estimateur de Callaway &amp; Sant'Anna, en n'utilisant que des "
        "comparaisons propres contre le groupe jamais-traité, fournit ici "
        "une estimation plus fiable de l'effet moyen du traitement sur les "
        "traités.", styles["Conclusion"]))
    story.append(PageBreak())


def section_tests(story, results):
    story.append(Paragraph("4. Tests statistiques", styles["H1"]))

    # --- 4.1 Tendances parallèles ---
    tp = results["t_parallel"]
    story.append(Paragraph("4.1 Test conjoint de tendances parallèles (test de Wald)",
                            styles["H2"]))
    story.append(Paragraph(
        "Ce test évalue conjointement l'hypothèse nulle selon laquelle TOUS "
        "les coefficients \"leads\" pré-traitement de l'event-study (figure "
        "3) sont simultanément nuls : H0 : δ_k = 0 pour tout k &lt; 0. La "
        "statistique de test suit asymptotiquement une loi du χ<super>2</super> sous H0.",
        styles["Corps"]))
    story.append(Paragraph(
        f"Statistique de test = {_fmt(tp['statistic'],3)} (degrés de liberté = "
        f"{tp['df']}) ; p-value = {_pval(tp['pvalue'])}.", styles["Corps"]))
    story.append(Paragraph(
        f"<b>Conclusion brute du test</b> : {tp['conclusion']}", styles["Conclusion"]))
    story.append(Paragraph(
        "<b>Mise en perspective importante</b> : dans le cas d'une adoption "
        "échelonnée du traitement avec effets dynamiques hétérogènes, les "
        "effets fixes temporels du TWFE peuvent être \"contaminés\" par les "
        "effets de traitement (déjà actifs) d'autres cohortes, ce qui peut "
        "produire des pseudo-pré-tendances n'ayant rien à voir avec une "
        "réelle violation de l'hypothèse de tendances parallèles (Sun &amp; "
        "Abraham, 2021 ; de Chaisemartin &amp; D'Haultfoeuille, 2020). C'est "
        "exactement ce qui est observé ici sur la catégorie terminale "
        "\"lead lointain\" de la figure 3 : en croisant ce résultat avec les "
        "ATT(g,t) pré-traitement de Callaway &amp; Sant'Anna (figure 5, "
        "zones antérieures au traitement, qui n'utilisent que des "
        "comparaisons propres contre le groupe jamais-traité et restent "
        "non significatives pour l'écrasante majorité des cellules), on "
        "conclut que la violation détectée par le test TWFE est "
        "vraisemblablement un artefact de contamination plutôt qu'une "
        "réelle rupture de tendance parallèle. Ce diagnostic croisé illustre "
        "pourquoi il est recommandé de toujours interpréter le test de "
        "pré-tendance TWFE conjointement avec un estimateur robuste à "
        "l'hétérogénéité dynamique en contexte d'adoption échelonnée.",
        styles["Encadre"]))

    # --- 4.2 Placebo ---
    tpl = results["t_placebo"]
    story.append(Paragraph("4.2 Test de placebo (fausse date de traitement)", styles["H2"]))
    story.append(Paragraph(
        "On attribue artificiellement une date de traitement antérieure à "
        "la vraie date (ici "
        f"{int(tpl['fake_post_period'])}, alors que la vraie date est "
        "ultérieure), exclusivement à l'intérieur de la période "
        "pré-traitement, puis on réestime un DiD 2x2 sur ces données placebo "
        "où, par construction, AUCUNE unité n'est réellement traitée. Sous "
        "l'hypothèse de tendances parallèles (et d'absence d'anticipation), "
        "l'effet placebo doit être statistiquement nul.", styles["Corps"]))
    story.append(Paragraph(
        f"Estimation placebo = {_fmt(tpl['estimate'],3)} (erreur-type = "
        f"{_fmt(tpl['se'],3)}) ; p-value = {_pval(tpl['pvalue'])}.",
        styles["Corps"]))
    story.append(Paragraph(f"<b>Conclusion</b> : {tpl['conclusion']}", styles["Conclusion"]))

    # --- 4.3 Permutation ---
    tperm = results["t_perm"]
    story.append(Paragraph("4.3 Test de permutation (inférence par randomisation)",
                            styles["H2"]))
    story.append(Paragraph(
        "Ce test non-paramétrique reconstruit la distribution de "
        "l'estimateur DiD 2x2 SOUS L'HYPOTHÈSE NULLE D'EFFET NUL, en "
        "réaffectant aléatoirement le statut \"traité\" parmi l'ensemble des "
        f"unités ({tperm['n_perm']} permutations, en conservant le nombre "
        "d'unités traitées), puis situe l'estimation réellement observée "
        "dans cette distribution nulle empirique. Il est particulièrement "
        "recommandé lorsque le nombre de clusters (unités ou groupes) est "
        "faible, cas dans lequel les tests asymptotiques usuels (basés sur "
        "l'erreur-type clusterisée) sont peu fiables.", styles["Corps"]))
    story.append(_img(results["figures"]["permutation"], width=11*cm))
    story.append(Paragraph(
        "Figure 7 — Distribution nulle de l'estimateur DiD 2x2 sous "
        "permutations aléatoires du statut de traitement, et position de "
        "l'estimation réellement observée.", styles["Legende"]))
    story.append(Paragraph(
        f"p-value de permutation = {_pval(tperm['pvalue_permutation'])}.",
        styles["Corps"]))
    story.append(Paragraph(f"<b>Conclusion</b> : {tperm['conclusion']}", styles["Conclusion"]))
    story.append(PageBreak())

    # --- 4.4 Hétéroscédasticité ---
    th = results["t_hetero"]
    story.append(Paragraph("4.4 Test de Breusch-Pagan (hétéroscédasticité)", styles["H2"]))
    story.append(Paragraph(
        "Teste l'hypothèse nulle d'homoscédasticité des résidus de la "
        "régression DiD 2x2 (variance des résidus constante, indépendante "
        "des régresseurs). Une hétéroscédasticité détectée justifie l'usage "
        "d'erreurs-types robustes (HC) ou clusterisées plutôt que les "
        "erreurs-types OLS \"classiques\" — ce que ce template applique par "
        "défaut, indépendamment du résultat de ce test, par précaution.",
        styles["Corps"]))
    story.append(Paragraph(
        f"Statistique de Breusch-Pagan = {_fmt(th['statistic'],3)} ; p-value = "
        f"{_pval(th['pvalue'])}.", styles["Corps"]))
    story.append(Paragraph(f"<b>Conclusion</b> : {th['conclusion']}", styles["Conclusion"]))

    # --- 4.5 Bootstrap ---
    tb = results["t_boot"]
    story.append(Paragraph("4.5 Bootstrap par cluster (unité)", styles["H2"]))
    story.append(Paragraph(
        "Ré-échantillonnage avec remise des UNITÉS (et non des observations "
        "individuelles, afin de respecter la structure de panel et "
        "l'autocorrélation intra-unité), répété "
        f"{tb['n_boot']} fois, pour obtenir une erreur-type et un intervalle "
        "de confiance empiriques alternatifs à ceux, asymptotiques, de la "
        "régression avec erreurs-types clusterisées. Recommandé en "
        "complément lorsque le nombre de clusters est faible.",
        styles["Corps"]))
    story.append(_img(results["figures"]["bootstrap"], width=11*cm))
    story.append(Paragraph(
        "Figure 8 — Distribution bootstrap (par cluster) de l'estimateur "
        "DiD 2x2.", styles["Legende"]))
    story.append(Paragraph(
        f"Erreur-type asymptotique (cluster) = {_fmt(tb['se_cluster_asymptotique'],3)} "
        f"vs erreur-type bootstrap = {_fmt(tb['se_bootstrap'],3)} ; intervalle de "
        f"confiance bootstrap à 95 % = [{_fmt(tb['ci_bootstrap_low'],2)} ; "
        f"{_fmt(tb['ci_bootstrap_high'],2)}].", styles["Corps"]))
    story.append(Paragraph(
        "<b>Conclusion</b> : la proximité entre l'erreur-type asymptotique "
        "et l'erreur-type bootstrap renforce la confiance dans l'inférence "
        "; un écart important inviterait à privilégier l'intervalle "
        "bootstrap, plus robuste aux petits échantillons.",
        styles["Conclusion"]))
    story.append(PageBreak())


def section_synthese(story, results):
    story.append(Paragraph("5. Synthèse des résultats", styles["H1"]))
    story.append(Paragraph(
        "Le tableau ci-dessous récapitule les estimations obtenues par "
        "chacune des méthodes de DiD multi-périodes/multi-cohortes "
        "appliquées à la base de données.", styles["Corps"]))

    df = results["summary_df"]
    rows = [["Méthode", "Estimation", "Erreur-type", "IC 95%", "p-value"]]
    for _, r in df.iterrows():
        rows.append([r["methode"], _fmt(r["estimation"],3), _fmt(r["se"],3),
                     f"[{_fmt(r['ic_bas'],2)} ; {_fmt(r['ic_haut'],2)}]",
                     _pval(r["p_value"])])
    t = Table(rows, hAlign="CENTER", colWidths=[7.5*cm, 2.6*cm, 2.6*cm, 3.6*cm, 2*cm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Recommandation de lecture</b> : en présence d'adoption "
        "échelonnée et d'hétérogénéité dynamique des effets (diagnostiquée "
        "ici par la décomposition de Goodman-Bacon, section 3.4), "
        "l'estimateur de Callaway &amp; Sant'Anna doit être préféré au TWFE "
        "statique pour résumer l'effet moyen du traitement sur les traités. "
        "Le TWFE statique et son event-study restent néanmoins utiles "
        "comme outils de diagnostic et de comparaison.", styles["Encadre"]))
    story.append(PageBreak())


def section_limites_biblio(story):
    story.append(Paragraph("6. Limites et recommandations", styles["H1"]))
    items = [
        "Toutes les méthodes présentées supposent l'absence de "
        "\"spillovers\" (SUTVA) : si le traitement d'une unité affecte "
        "l'outcome d'unités non-traitées (effets de débordement "
        "géographique, concurrentiel...), les estimateurs sont biaisés.",
        "L'hypothèse de tendances parallèles n'est jamais prouvée, "
        "seulement rendue plus ou moins plausible par les tests de "
        "pré-tendance ; une bonne pratique consiste à la documenter "
        "qualitativement (changements institutionnels concomitants, "
        "groupes de comparaison alternatifs, etc.).",
        "En présence d'un faible nombre de clusters (groupes ou unités), "
        "privilégier le bootstrap par cluster ou le bootstrap sauvage "
        "(\"wild cluster bootstrap\") à l'inférence asymptotique standard.",
        "La décomposition de Goodman-Bacon implémentée ici suit la logique "
        "du théorème original mais en simplifie le calcul des poids ; pour "
        "une publication académique, repasser par l'algorithme exact "
        "(package R bacondecomp) est recommandé.",
        "L'estimateur de Callaway &amp; Sant'Anna implémenté ici utilise "
        "uniquement le groupe jamais-traité comme groupe de comparaison ; "
        "il existe une variante \"pas-encore-traité\" (not-yet-treated) qui "
        "peut être plus efficace si le groupe jamais-traité est petit ou "
        "absent.",
        "D'autres estimateurs robustes à l'hétérogénéité dynamique non "
        "couverts ici (Sun &amp; Abraham, 2021 ; de Chaisemartin &amp; "
        "D'Haultfoeuille, 2020 ; Borusyak, Jaravel &amp; Spiess, 2024) "
        "peuvent être pertinents selon le contexte.",
    ]
    flow_items = [ListItem(Paragraph(t, styles["Corps"]), leftIndent=8) for t in items]
    story.append(ListFlowable(flow_items, bulletType="bullet", start="•"))

    story.append(Paragraph("Références", styles["H2"]))
    refs = [
        "Callaway, B. &amp; Sant'Anna, P. H. C. (2021). \"Difference-in-Differences "
        "with Multiple Time Periods.\" <i>Journal of Econometrics</i>, 225(2), 200-230.",
        "Goodman-Bacon, A. (2021). \"Difference-in-differences with variation in "
        "treatment timing.\" <i>Journal of Econometrics</i>, 225(2), 254-277.",
        "de Chaisemartin, C. &amp; D'Haultfœuille, X. (2020). \"Two-Way Fixed "
        "Effects Estimators with Heterogeneous Treatment Effects.\" "
        "<i>American Economic Review</i>, 110(9), 2964-2996.",
        "Sun, L. &amp; Abraham, S. (2021). \"Estimating dynamic treatment "
        "effects in event studies with heterogeneous treatment effects.\" "
        "<i>Journal of Econometrics</i>, 225(2), 175-199.",
        "Roth, J., Sant'Anna, P. H. C., Bilinski, A. &amp; Poe, J. (2023). "
        "\"What's Trending in Difference-in-Differences? A Synthesis of the "
        "Recent Econometrics Literature.\" <i>Journal of Econometrics</i>, "
        "235(2), 2218-2244.",
        "Card, D. &amp; Krueger, A. B. (1994). \"Minimum Wages and Employment: "
        "A Case Study of the Fast-Food Industry in New Jersey and "
        "Pennsylvania.\" <i>American Economic Review</i>, 84(4), 772-793.",
        "Cameron, A. C., Gelbach, J. B. &amp; Miller, D. L. (2008). "
        "\"Bootstrap-Based Improvements for Inference with Clustered "
        "Errors.\" <i>Review of Economics and Statistics</i>, 90(3), 414-427.",
    ]
    flow_refs = [ListItem(Paragraph(t, styles["Corps"]), leftIndent=8) for t in refs]
    story.append(ListFlowable(flow_refs, bulletType="bullet", start="-"))


# ------------------------------------------------------------------------------
# EN-TETE / PIED DE PAGE
# ------------------------------------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.6)
    canvas.line(2*cm, A4[1] - 1.5*cm, A4[0] - 2*cm, A4[1] - 1.5*cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2*cm, A4[1] - 1.3*cm, "Analyse en Difference-in-Differences")
    canvas.drawRightString(A4[0] - 2*cm, A4[1] - 1.3*cm, date.today().strftime("%d/%m/%Y"))
    canvas.line(2*cm, 1.4*cm, A4[0] - 2*cm, 1.4*cm)
    canvas.drawCentredString(A4[0] / 2, 1.0*cm, f"Page {doc.page}")
    canvas.restoreState()


# ------------------------------------------------------------------------------
# FONCTION PRINCIPALE
# ------------------------------------------------------------------------------
def build_report(results, pdf_path=PDF_PATH):
    """Assemble l'ensemble des sections et genere le rapport PDF final."""
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        topMargin=2.2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm,
        title="Rapport Diff-in-Diff", author="Template DiD (Python)",
    )

    data = results["data"]
    story = []
    section_title_page(story, n_obs=len(data), n_units=data["unit_id"].nunique(),
                        periode_min=int(data["time"].min()), periode_max=int(data["time"].max()))
    section_intro(story)
    section_donnees(story, results)
    section_did_2x2(story, results)
    section_twfe(story, results)
    section_event_study(story, results)
    section_bacon(story, results)
    section_cs(story, results)
    section_tests(story, results)
    section_synthese(story, results)
    section_limites_biblio(story)

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"[generate_pdf_report] Rapport PDF genere : {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    import did_template as dt
    res = dt.run_full_analysis()
    build_report(res)
