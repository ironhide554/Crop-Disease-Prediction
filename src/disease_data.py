"""
disease_data.py - Authoritative mapping of the 38 PlantVillage crop disease classes
along with detailed agronomic metadata, causes, symptoms, and remedies.
"""

CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Friendly display names mapped by index
CLASS_MAPPING = {
    0: 'Apple - Apple Scab',
    1: 'Apple - Black Rot',
    2: 'Apple - Cedar Apple Rust',
    3: 'Apple - Healthy',
    4: 'Blueberry - Healthy',
    5: 'Cherry (including sour) - Powdery Mildew',
    6: 'Cherry (including sour) - Healthy',
    7: 'Corn (maize) - Cercospora Leaf Spot / Gray Leaf Spot',
    8: 'Corn (maize) - Common Rust',
    9: 'Corn (maize) - Northern Leaf Blight',
    10: 'Corn (maize) - Healthy',
    11: 'Grape - Black Rot',
    12: 'Grape - Esca (Black Measles)',
    13: 'Grape - Leaf Blight (Isariopsis Leaf Spot)',
    14: 'Grape - Healthy',
    15: 'Orange - Huanglongbing (Citrus Greening)',
    16: 'Peach - Bacterial Spot',
    17: 'Peach - Healthy',
    18: 'Pepper Bell - Bacterial Spot',
    19: 'Pepper Bell - Healthy',
    20: 'Potato - Early Blight',
    21: 'Potato - Late Blight',
    22: 'Potato - Healthy',
    23: 'Raspberry - Healthy',
    24: 'Soybean - Healthy',
    25: 'Squash - Powdery Mildew',
    26: 'Strawberry - Leaf Scorch',
    27: 'Strawberry - Healthy',
    28: 'Tomato - Bacterial Spot',
    29: 'Tomato - Early Blight',
    30: 'Tomato - Late Blight',
    31: 'Tomato - Leaf Mold',
    32: 'Tomato - Septoria Leaf Spot',
    33: 'Tomato - Two-Spotted Spider Mite',
    34: 'Tomato - Target Spot',
    35: 'Tomato - Tomato Yellow Leaf Curl Virus',
    36: 'Tomato - Tomato Mosaic Virus',
    37: 'Tomato - Healthy'
}

DISEASE_DETAILS = {
    0: {
        "crop": "Apple",
        "disease": "Apple Scab",
        "category": "Fungal (Venturia inaequalis)",
        "symptoms": "Olive-green to velvety brown spots on leaves and fruit, causing leaf yellowing and premature leaf drop.",
        "organic_treatment": "Apply liquid copper fungicide or sulfur sprays in early spring. Remove and compost or burn fallen infected leaves.",
        "chemical_treatment": "Captan, Myclobutanil, or Mancozeb sprays applied during early bud break to petal fall.",
        "prevention": "Prune trees to improve canopy air circulation. Plant scab-resistant apple cultivars like 'Liberty' or 'Enterprise'."
    },
    1: {
        "crop": "Apple",
        "disease": "Black Rot",
        "category": "Fungal (Botryosphaeria obtusa)",
        "symptoms": "Circular purple-bordered leaf lesions ('frogeye'), sunken brown bark cankers, and black mummified fruit.",
        "organic_treatment": "Prune dead wood, remove mummified apples, and apply copper-based fungicides at dormancy break.",
        "chemical_treatment": "Fungicides containing Thiophanate-methyl, Captan, or Strobilurins.",
        "prevention": "Sanitize pruning tools between cuts. Keep trees vigorously healthy and prune away dead or fire-blighted limbs."
    },
    2: {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "category": "Fungal (Gymnosporangium juniperi-virginianae)",
        "symptoms": "Bright yellow-orange or reddish spots on upper leaf surfaces that develop tube-like fungal structures beneath.",
        "organic_treatment": "Neem oil or sulfur sprays applied when cedar galls release orange gelatinous horns in spring.",
        "chemical_treatment": "Myclobutanil or Chlorothalonil sprays starting at pink bud stage.",
        "prevention": "Remove nearby Eastern red cedar or juniper trees within several hundred yards of apple orchards."
    },
    3: {
        "crop": "Apple",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Foliage is vibrant green, firm, unblemished, and exhibiting active photosynthesis.",
        "organic_treatment": "Maintain balanced nitrogen-phosphorus-potassium organic fertilization and adequate mulch.",
        "chemical_treatment": "None needed. Maintain routine preventative monitoring.",
        "prevention": "Ensure good spacing, annual pruning, and consistent drip irrigation."
    },
    4: {
        "crop": "Blueberry",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Healthy deep green foliage with sturdy woody stems and normal berry cluster development.",
        "organic_treatment": "Maintain acidic soil pH (4.5 to 5.2) with peat moss or elemental sulfur.",
        "chemical_treatment": "None needed.",
        "prevention": "Provide consistent moisture without waterlogging and replenish pine bark mulch annually."
    },
    5: {
        "crop": "Cherry",
        "disease": "Powdery Mildew",
        "category": "Fungal (Podosphaera clandestina)",
        "symptoms": "White powdery fungal patches on new leaves, shoot distortion, and curled foliage.",
        "organic_treatment": "Apply potassium bicarbonate, neem oil, or dilute milk spray (1:9 with water) in direct sunlight.",
        "chemical_treatment": "Myclobutanil, Triflumizole, or sulfur-based fungicides.",
        "prevention": "Prune dense canopies to promote sunlight penetration and low relative humidity within the tree."
    },
    6: {
        "crop": "Cherry",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Lush green leaves, healthy bark, and normal blossoming and fruiting.",
        "organic_treatment": "Ensure well-draining soil and organic compost application.",
        "chemical_treatment": "None needed.",
        "prevention": "Regularly monitor for aphid activity and bird netting during fruit ripening."
    },
    7: {
        "crop": "Corn (Maize)",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "category": "Fungal (Cercospora zeae-maydis)",
        "symptoms": "Tan, rectangular lesions restricted between leaf veins with sharp parallel borders.",
        "organic_treatment": "Apply bio-fungicides containing Bacillus subtilis or copper hydroxide.",
        "chemical_treatment": "Foliar fungicides such as Pyraclostrobin, Azoxystrobin, or Propiconazole at tasseling stage.",
        "prevention": "Practice 2-year crop rotation out of corn; till or manage surface residue to promote decomposition."
    },
    8: {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "category": "Fungal (Puccinia sorghi)",
        "symptoms": "Small, powdery cinnamon-brown to golden-red pustules scattered across both leaf surfaces.",
        "organic_treatment": "Sulfur dusts or horticultural oils early in the season before extensive pustule coverage.",
        "chemical_treatment": "Fungicide mixtures containing Strobilurin + Triazole (e.g., Azoxystrobin + Difenoconazole).",
        "prevention": "Plant resistant hybrid corn varieties; plant early to avoid peak midsummer spore dispersal."
    },
    9: {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "category": "Fungal (Setosphaeria turcica)",
        "symptoms": "Long, elliptical cigar-shaped grayish-green to tan lesions that can measure 1 to 6 inches long.",
        "organic_treatment": "Copper fungicides and Bacillus amyloliquefaciens biocontrols.",
        "chemical_treatment": "Foliar applications of Pyraclostrobin, Metconazole, or Fluxapyroxad.",
        "prevention": "Deep tillage to bury infected debris, rotation to non-host crops like soybeans or alfalfa."
    },
    10: {
        "crop": "Corn (Maize)",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Broad, deep green leaves with sturdy stalks and uniform ear silk development.",
        "organic_treatment": "Maintain balanced nitrogen levels and side-dress compost.",
        "chemical_treatment": "None needed.",
        "prevention": "Scout weekly for armyworms and maintain optimal field drainage."
    },
    11: {
        "crop": "Grape",
        "disease": "Black Rot",
        "category": "Fungal (Guignardia bidwellii)",
        "symptoms": "Reddish-brown circular leaf spots with tiny black dots (pycnidia); berries shrivel into hard, black raisins.",
        "organic_treatment": "Copper or sulfur sprays starting at 10-inch shoot growth. Destroy infected cane debris.",
        "chemical_treatment": "Myclobutanil, Mancozeb, or Ziram before and after bloom.",
        "prevention": "Canopy thinning, leaf pulling around fruit clusters to maximize airflow and rapid drying."
    },
    12: {
        "crop": "Grape",
        "disease": "Esca (Black Measles)",
        "category": "Fungal Complex (Phaeomoniella, Fomitiporia)",
        "symptoms": "'Tiger-stripe' leaf discoloration with yellow and necrotic margins; spotted berries with dark purple spots.",
        "organic_treatment": "Treat pruning wounds with biological wound sealants (Trichoderma harzianum).",
        "chemical_treatment": "No curative chemical exists; trunk renewal or remedial surgery is standard.",
        "prevention": "Avoid pruning during damp, wet weather; sterilize pruning shears between vines."
    },
    13: {
        "crop": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "category": "Fungal (Pseudocercospora vitis)",
        "symptoms": "Irregular dark brown to black spots with yellowish halos on mature leaves, leading to defoliation.",
        "organic_treatment": "Bordeaux mixture (copper sulfate + slaked lime) sprayed after harvest and in early spring.",
        "chemical_treatment": "Chlorothalonil or Mancozeb sprays.",
        "prevention": "Remove post-harvest fallen leaf litter and improve trellis air circulation."
    },
    14: {
        "crop": "Grape",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Vigorous green vine foliage, strong tendrils, and clean developing clusters.",
        "organic_treatment": "Balanced composting and drip irrigation at the root zone.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain vine training and canopy management."
    },
    15: {
        "crop": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "category": "Bacterial (Candidatus Liberibacter asiaticus)",
        "symptoms": "Asymmetrical blotchy leaf mottle, yellow veins, small lopsided bitter fruit that stays green at base.",
        "organic_treatment": "Strict vector control: neem oil or insecticidal soap against Asian Citrus Psyllid; foliar micronutrient feeding.",
        "chemical_treatment": "Systemic insecticides (Imidacloprid) to suppress psyllid populations; antibiotic trunk injections where permitted.",
        "prevention": "Plant only certified disease-free nursery rootstock; immediately eradicate confirmed infected trees."
    },
    16: {
        "crop": "Peach",
        "disease": "Bacterial Spot",
        "category": "Bacterial (Xanthomonas arboricola pv. pruni)",
        "symptoms": "Small, angular purple-brown spots on leaves that drop out creating a 'shot-hole' look; cracked fruit.",
        "organic_treatment": "Fixed copper sprays during dormancy and early shuck split; avoid high pressure blasting.",
        "chemical_treatment": "Oxytetracycline sprays applied during high disease pressure warm spring rains.",
        "prevention": "Choose resistant peach varieties (e.g. 'Biscoe', 'Candor'); avoid excessive high-nitrogen fertilizers."
    },
    17: {
        "crop": "Peach",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Smooth, lanceolate leaves, sturdy shoots, and healthy velvet-skinned fruit.",
        "organic_treatment": "Maintain open center canopy pruning and compost tea soil conditioning.",
        "chemical_treatment": "None needed.",
        "prevention": "Dormant copper sprays against leaf curl as regular seasonal maintenance."
    },
    18: {
        "crop": "Pepper Bell",
        "disease": "Bacterial Spot",
        "category": "Bacterial (Xanthomonas campestris pv. vesicatoria)",
        "symptoms": "Water-soaked lesions that turn dark brown with yellow rings; rough raised scabs on pepper fruit.",
        "organic_treatment": "Copper hydroxide mixed with biologicals (Bacillus subtilis). Avoid working in wet fields.",
        "chemical_treatment": "Copper + Mancozeb tank mixes for improved bacterial suppression.",
        "prevention": "Use hot-water treated certified seeds, sanitize stakes, and maintain 3-year solanaceous crop rotation."
    },
    19: {
        "crop": "Pepper Bell",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Crisp green foliage, sturdy central stems, and uniform firm bell fruit.",
        "organic_treatment": "Provide calcium-rich organic fertilizer to prevent blossom-end rot.",
        "chemical_treatment": "None needed.",
        "prevention": "Consistent watering and weed mulching."
    },
    20: {
        "crop": "Potato",
        "disease": "Early Blight",
        "category": "Fungal (Alternaria solani)",
        "symptoms": "Dark brown circular spots with distinctive concentric rings ('target board' pattern) on older lower leaves.",
        "organic_treatment": "Copper octanoate or copper sulfate sprays; remove diseased lower foliage.",
        "chemical_treatment": "Chlorothalonil, Mancozeb, or Azoxystrobin applied every 7-10 days under warm, humid conditions.",
        "prevention": "Maintain proper vine nutrition; avoid overhead sprinkler irrigation; space plants properly."
    },
    21: {
        "crop": "Potato",
        "disease": "Late Blight",
        "category": "Oomycete (Phytophthora infestans)",
        "symptoms": "Fast-spreading water-soaked dark lesions on leaves with white fuzzy fungal growth on leaf undersides in high humidity.",
        "organic_treatment": "Preventative copper fungicide applied thoroughly before disease onset; immediately destroy infected vines.",
        "chemical_treatment": "Mandipropamid, Fluopicolide, or Cymoxanil; highly aggressive management needed.",
        "prevention": "Plant certified disease-free seed tubers, eliminate cull piles, and destroy volunteer potato plants."
    },
    22: {
        "crop": "Potato",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Lush compound green leaves, robust foliage canopy, and sturdy root system.",
        "organic_treatment": "Hilling with compost/straw and balanced potassium feeding.",
        "chemical_treatment": "None needed.",
        "prevention": "Scout for Colorado potato beetles and maintain consistent soil moisture."
    },
    23: {
        "crop": "Raspberry",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Vibrant serrated green leaves, clean canes, and healthy berry clusters.",
        "organic_treatment": "Organic bone meal and compost; maintain weed-free rows with wood shavings.",
        "chemical_treatment": "None needed.",
        "prevention": "Annual pruning of floricanes after fruiting to maintain ventilation."
    },
    24: {
        "crop": "Soybean",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Deep green trifoliate leaves, strong nodulated roots, and well-filled seed pods.",
        "organic_treatment": "Inoculate seeds with Rhizobium bacteria for natural nitrogen fixation.",
        "chemical_treatment": "None needed.",
        "prevention": "Maintain crop rotation with corn or small grains."
    },
    25: {
        "crop": "Squash",
        "disease": "Powdery Mildew",
        "category": "Fungal (Podosphaera xanthii)",
        "symptoms": "White talcum-powder-like fungal patches on leaf surfaces and petioles; leaves turn yellow and brittle.",
        "organic_treatment": "Potassium bicarbonate spray, neem oil, sulfur sprays, or 1:9 milk-water solution.",
        "chemical_treatment": "Trifloxystrobin, Myclobutanil, or Quinoxyfen.",
        "prevention": "Select powdery-mildew resistant squash varieties; plant in full sun with wide plant spacing."
    },
    26: {
        "crop": "Strawberry",
        "disease": "Leaf Scorch",
        "category": "Fungal (Diplocarpon earlianum)",
        "symptoms": "Numerous small, dark purple irregular blotches that enlarge until entire leaves appear burned or scorched.",
        "organic_treatment": "Copper soaps applied after post-harvest renovation; prune diseased leaves.",
        "chemical_treatment": "Captan, Thiophanate-methyl, or Pyraclostrobin.",
        "prevention": "Renovate strawberry beds annually, remove older leaves, and use straw mulch to prevent soil splashing."
    },
    27: {
        "crop": "Strawberry",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Vibrant trifoliate green leaves, white flowers, and firm sweet developing berries.",
        "organic_treatment": "Pine needle or clean straw mulch; organic seaweed extract foliar spray.",
        "chemical_treatment": "None needed.",
        "prevention": "Drip irrigation under mulch to keep foliage and crowns completely dry."
    },
    28: {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "category": "Bacterial (Xanthomonas perforans / euvesicatoria)",
        "symptoms": "Small, dark greasy spots with yellow halos on leaves; small raised blister-like spots on tomatoes.",
        "organic_treatment": "Liquid copper sprays combined with biological bio-fungicide (Bacillus subtilis).",
        "chemical_treatment": "Copper hydroxide mixed with Mancozeb for enhanced bactericidal action.",
        "prevention": "Avoid overhead watering; disinfect garden stakes and cages; rotate away from nightshades for 3 years."
    },
    29: {
        "crop": "Tomato",
        "disease": "Early Blight",
        "category": "Fungal (Alternaria solani)",
        "symptoms": "Concentric ring 'bullseye' dark spots starting on bottom leaves, yellowing surrounding tissue and causing defoliation.",
        "organic_treatment": "Prune bottom 12 inches of leaves; apply copper fungicide or Serenade biofungicide.",
        "chemical_treatment": "Chlorothalonil, Mancozeb, or Difenoconazole at the first sign of lower leaf spotting.",
        "prevention": "Heavy mulch beneath plants to prevent rain splash from soil; stake and cage plants for vertical airflow."
    },
    30: {
        "crop": "Tomato",
        "disease": "Late Blight",
        "category": "Oomycete (Phytophthora infestans)",
        "symptoms": "Rapidly expanding dark water-soaked greasy patches on leaves and stems; white mold underneath during cool, damp weather.",
        "organic_treatment": "Fixed copper preventative spray; immediately pull and bag entire infected plants to prevent airborne spread.",
        "chemical_treatment": "Mandipropamid, Chlorothalonil, or Cymoxanil sprayed preventatively.",
        "prevention": "Ensure good spacing; destroy volunteer tomato and potato plants; avoid working with wet plants."
    },
    31: {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "category": "Fungal (Passalora fulva)",
        "symptoms": "Pale greenish-yellow spots on upper leaf surfaces with olive-green to brown velvety fungal patches underneath.",
        "organic_treatment": "Copper fungicide; significantly reduce greenhouse relative humidity below 85%.",
        "chemical_treatment": "Chlorothalonil or Mancozeb sprays.",
        "prevention": "Install exhaust fans in greenhouse/high tunnel; prune lower leaves to promote good bottom ventilation."
    },
    32: {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "category": "Fungal (Septoria lycopersici)",
        "symptoms": "Dozens of small circular spots with gray or tan centers and dark brown borders, dotted with tiny black fruiting bodies.",
        "organic_treatment": "Copper fungicide applied every 7 days; remove lower infected leaves immediately.",
        "chemical_treatment": "Chlorothalonil, Mancozeb, or Quadris (Azoxystrobin).",
        "prevention": "Do not overhead water; apply 2-3 inches of clean organic straw mulch around plant base."
    },
    33: {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mite",
        "category": "Pest / Arachnid (Tetranychus urticae)",
        "symptoms": "Fine yellow or white stippling on leaf tops; delicate silken webbing on leaf undersides and leaf drop.",
        "organic_treatment": "Insecticidal soap, neem oil, or rosemary oil spray; release beneficial predatory mites (Phytoseiulus persimilis).",
        "chemical_treatment": "Miticide sprays (Abamectin, Bifenazate, or Spiromesifen) ensuring coverage under leaves.",
        "prevention": "Keep plants adequately watered (spider mites flourish in hot, dusty, drought-stressed conditions)."
    },
    34: {
        "crop": "Tomato",
        "disease": "Target Spot",
        "category": "Fungal (Corynespora cassiicola)",
        "symptoms": "Brown circular lesions with faint concentric rings; center often cracks or tears; leaves yellow and shed.",
        "organic_treatment": "Copper fungicides and bio-fungicides; remove severely diseased leaves.",
        "chemical_treatment": "Boscalid, Pyraclostrobin, or Chlorothalonil.",
        "prevention": "Wide plant spacing; prune suckers to increase airflow; eliminate broadleaf weeds around field borders."
    },
    35: {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "category": "Viral (Begomovirus transmitted by Whiteflies)",
        "symptoms": "Severe upward leaf curling, yellow leaf margins, stunting of plants, and flower abortion.",
        "organic_treatment": "No cure for virus; control whitefly vectors using yellow sticky traps, insecticidal soaps, and reflective mulches.",
        "chemical_treatment": "Systemic insecticides (Dinotefuran, Imidacloprid) to target whiteflies.",
        "prevention": "Grow TYLCV-resistant hybrids; install 50-mesh insect netting in greenhouses; destroy infected crops promptly."
    },
    36: {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "category": "Viral (Tobamovirus)",
        "symptoms": "Mottled light and dark green mosaic patterns on foliage, leaf distortion ('shoestringing'), and internal fruit browning.",
        "organic_treatment": "No cure exists; remove and destroy infected plants immediately. Dip tools in 20% nonfat dry milk solution.",
        "chemical_treatment": "None effective against viruses.",
        "prevention": "Do not smoke or use tobacco near tomato plants (tobacco carries virus); sanitize tools and wash hands thoroughly."
    },
    37: {
        "crop": "Tomato",
        "disease": "Healthy",
        "category": "Healthy Crop",
        "symptoms": "Deep green compound leaves, sturdy hairy stems, bright yellow blossoms, and smooth uniform tomatoes.",
        "organic_treatment": "Maintain balanced organic tomato fertilizer (high in phosphorus and calcium) and consistent watering.",
        "chemical_treatment": "None needed.",
        "prevention": "Prune suckers, mulch well, stake securely, and rotate crops annually."
    }
}

def get_disease_info(class_index: int) -> dict:
    """Returns detailed agronomic information for the given class index."""
    if class_index in DISEASE_DETAILS:
        info = DISEASE_DETAILS[class_index].copy()
        info['class_name'] = CLASS_MAPPING.get(class_index, 'Unknown')
        info['raw_class_name'] = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else 'Unknown'
        return info
    return {
        "crop": "Unknown",
        "disease": "Unknown Disease",
        "category": "Unclassified",
        "symptoms": "No symptom data available.",
        "organic_treatment": "Consult local agricultural extension.",
        "chemical_treatment": "Consult local agricultural extension.",
        "prevention": "Maintain routine crop monitoring and clean cultivation practices.",
        "class_name": f"Class {class_index}",
        "raw_class_name": f"class_{class_index}"
    }
