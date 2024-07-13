# disease_checker/utils.py

def predict_disease(data):
    # Leaf color related conditions
    if data['leaf_color'] == 'yellow':
        if data['wilting']:
            if data['soil_type'] == 'sandy' and data['watering_frequency'] in ['monthly', 'biweekly']:
                return "Possible Disease: Drought Stress"
            if data['soil_type'] == 'clay' and data['weather_conditions'] == 'rainy':
                return "Possible Disease: Waterlogging"
            return "Possible Disease: Nutrient Deficiency (Nitrogen)"
        if data['spots_on_leaves']:
            return "Possible Disease: Bacterial Leaf Spot"
        if data['pest_presence']:
            return "Possible Disease: Aphid Infestation"
        return "Possible Disease: Iron Chlorosis"
    
    if data['leaf_color'] == 'brown':
        if data['spots_on_leaves']:
            if data['pest_presence']:
                return "Possible Disease: Leaf Spot (Caused by Pests)"
            return "Possible Disease: Fungal Infection (Septoria Leaf Spot)"
        if data['wilting']:
            if data['stem_condition'] == 'soft':
                return "Possible Disease: Root Rot"
            return "Possible Disease: Sunscald"
        if data['weather_conditions'] == 'windy' and data['stem_condition'] == 'brittle':
            return "Possible Disease: Wind Burn"
    
    if data['leaf_color'] == 'black':
        if data['mold_growth']:
            return "Possible Disease: Sooty Mold"
        if data['wilting']:
            return "Possible Disease: Frost Damage"
        if data['spots_on_leaves']:
            return "Possible Disease: Black Spot Disease"
    
    if data['leaf_color'] == 'purple':
        if data['wilting']:
            return "Possible Disease: Phosphorus Deficiency"
        if data['spots_on_leaves']:
            return "Possible Disease: Purple Blotch"
    
    # Leaf shape related conditions
    if data['leaf_shape'] == 'heart':
        if data['leaf_color'] == 'yellow' and data['wilting']:
            return "Possible Disease: Verticillium Wilt"
    
    if data['leaf_shape'] == 'lance' and data['spots_on_leaves'] and data['mold_growth']:
        return "Possible Disease: Rust Disease"
    
    # Stem condition related conditions
    if data['stem_condition'] == 'discolored':
        if data['leaf_color'] == 'brown' or data['leaf_color'] == 'black':
            return "Possible Disease: Stem Canker"
        return "Possible Disease: General Stem Rot"
    
    if data['stem_condition'] == 'brittle' and data['wilting']:
        return "Possible Disease: Boron Deficiency"
    
    # Mold growth related conditions
    if data['mold_growth']:
        if data['weather_conditions'] == 'rainy' and data['soil_type'] in ['clay', 'silty']:
            return "Possible Disease: Downy Mildew"
        return "Possible Disease: Powdery Mildew"
    
    # Pest presence related conditions
    if data['pest_presence']:
        if data['spots_on_leaves']:
            return "Possible Disease: Pest Infestation (Spider Mites)"
        if data['leaf_color'] == 'yellow':
            return "Possible Disease: Pest Infestation (Aphids)"
    
    # Crop-specific conditions
    if data['crop_type'].lower() == 'tomato':
        if data['leaf_color'] == 'yellow' and data['spots_on_leaves']:
            return "Possible Disease: Early Blight"
        if data['leaf_color'] == 'brown' and data['spots_on_leaves']:
            return "Possible Disease: Late Blight"
        if data['leaf_color'] == 'yellow' and data['mold_growth']:
            return "Possible Disease: Tomato Yellow Leaf Curl Virus"
    
    if data['crop_type'].lower() == 'potato':
        if data['leaf_color'] == 'yellow' and data['spots_on_leaves']:
            return "Possible Disease: Potato Virus Y"
        if data['leaf_color'] == 'brown' and data['spots_on_leaves']:
            return "Possible Disease: Alternaria Leaf Spot"
        if data['stem_condition'] == 'soft' and data['wilting']:
            return "Possible Disease: Potato Blackleg"
    
    # General fallback condition
    return "Unknown Disease: Please consult an expert."