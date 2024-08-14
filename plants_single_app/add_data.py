import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'plants_single_app.settings')  # Replace with your project's settings module
django.setup()

from core.models import Disease, FAQGroup, FAQItem

# Data for Disease model
diseases = [
    {
        'name': 'Powdery Mildew',
        'symptoms': 'White powdery spots on leaves and stems.',
        'description': 'Powdery mildew is a fungal disease that affects a wide range of plants.',
        'prevention': 'Ensure good air circulation, avoid overhead watering, and use resistant varieties.',
        'treatment': 'Apply fungicides, remove affected leaves, and improve air circulation.',
        'image': 'https://example.com/images/powdery_mildew.jpg',
    },
    {
        'name': 'Downy Mildew',
        'symptoms': 'Yellow or white patches on the upper surface of leaves.',
        'description': 'Downy mildew is a fungal disease that causes leaf spots and blights.',
        'prevention': 'Use resistant varieties, rotate crops, and avoid overhead irrigation.',
        'treatment': 'Apply fungicides, remove infected plants, and ensure good air circulation.',
        'image': 'https://example.com/images/downy_mildew.jpg',
    },
    {
        'name': 'Late Blight',
        'symptoms': 'Dark brown or black lesions on leaves and stems.',
        'description': 'Late blight is a serious disease that affects potatoes and tomatoes.',
        'prevention': 'Use resistant varieties, rotate crops, and avoid overhead watering.',
        'treatment': 'Apply fungicides, remove affected plants, and improve air circulation.',
        'image': 'https://example.com/images/late_blight.jpg',
    },
    {
        'name': 'Early Blight',
        'symptoms': 'Small brown spots on leaves, often with concentric rings.',
        'description': 'Early blight is a common disease that affects tomatoes and potatoes.',
        'prevention': 'Use resistant varieties, practice crop rotation, and maintain plant health.',
        'treatment': 'Apply fungicides, remove affected leaves, and improve air circulation.',
        'image': 'https://example.com/images/early_blight.jpg',
    },
    {
        'name': 'Rust',
        'symptoms': 'Orange or yellow pustules on leaves and stems.',
        'description': 'Rust is a fungal disease that affects a wide range of plants.',
        'prevention': 'Use resistant varieties, practice crop rotation, and avoid overhead watering.',
        'treatment': 'Apply fungicides, remove affected leaves, and improve air circulation.',
        'image': 'https://example.com/images/rust.jpg',
    },
    {
        'name': 'Anthracnose',
        'symptoms': 'Dark, sunken lesions on leaves, stems, and fruit.',
        'description': 'Anthracnose is a fungal disease that affects many different plants.',
        'prevention': 'Use resistant varieties, avoid overhead watering, and maintain plant health.',
        'treatment': 'Apply fungicides, remove affected leaves, and improve air circulation.',
        'image': 'https://example.com/images/anthracnose.jpg',
    },
    {
        'name': 'Fusarium Wilt',
        'symptoms': 'Yellowing and wilting of leaves, often starting on one side of the plant.',
        'description': 'Fusarium wilt is a soil-borne fungal disease that affects many plants.',
        'prevention': 'Use resistant varieties, practice crop rotation, and improve soil drainage.',
        'treatment': 'Remove and destroy affected plants, and improve soil health.',
        'image': 'https://example.com/images/fusarium_wilt.jpg',
    },
    {
        'name': 'Verticillium Wilt',
        'symptoms': 'Yellowing and wilting of leaves, often starting on one side of the plant.',
        'description': 'Verticillium wilt is a soil-borne fungal disease that affects many plants.',
        'prevention': 'Use resistant varieties, practice crop rotation, and improve soil drainage.',
        'treatment': 'Remove and destroy affected plants, and improve soil health.',
        'image': 'https://example.com/images/verticillium_wilt.jpg',
    },
    {
        'name': 'Botrytis Blight',
        'symptoms': 'Gray mold on leaves, stems, and flowers.',
        'description': 'Botrytis blight is a fungal disease that affects many plants.',
        'prevention': 'Ensure good air circulation, avoid overhead watering, and remove affected plant parts.',
        'treatment': 'Apply fungicides, remove affected leaves, and improve air circulation.',
        'image': 'https://example.com/images/botrytis_blight.jpg',
    },
    {
        'name': 'Black Spot',
        'symptoms': 'Black spots on leaves, often with yellow halos.',
        'description': 'Black spot is a fungal disease that primarily affects roses.',
        'prevention': 'Use resistant varieties, avoid overhead watering, and maintain plant health.',
        'treatment': 'Apply fungicides, remove affected leaves, and improve air circulation.',
        'image': 'https://example.com/images/black_spot.jpg',
    }
]

# Data for FAQGroup and FAQItem models
faq_data = {
    'General': [
        {
            'question': 'What is plant disease?',
            'answer': 'Plant disease is an impairment of the normal state of a plant that interrupts or modifies its vital functions.'
        },
        {
            'question': 'How do I prevent plant diseases?',
            'answer': 'You can prevent plant diseases by practicing good hygiene, using disease-resistant varieties, and applying proper cultural practices.'
        },
        {
            'question': 'What are the common signs of plant disease?',
            'answer': 'Common signs include spots on leaves, wilting, mold growth, and discolored stems.'
        },
        {
            'question': 'How do I know if my plant is diseased?',
            'answer': 'Look for symptoms such as abnormal growth, discoloration, wilting, and the presence of spots or mold.'
        },
        {
            'question': 'Can plant diseases spread to other plants?',
            'answer': 'Yes, many plant diseases can spread through water, air, soil, or direct contact with infected plants.'
        },
    ],
    'Treatment': [
        {
            'question': 'How do I treat powdery mildew?',
            'answer': 'You can treat powdery mildew by applying fungicides, removing affected leaves, and improving air circulation.'
        },
        {
            'question': 'What should I do if my plant has downy mildew?',
            'answer': 'Apply fungicides, remove infected plants, and ensure good air circulation to treat downy mildew.'
        },
        {
            'question': 'How do I treat late blight?',
            'answer': 'Apply fungicides, remove affected plants, and avoid overhead watering to treat late blight.'
        },
        {
            'question': 'What is the best way to treat rust on plants?',
            'answer': 'Apply fungicides, remove affected leaves, and improve air circulation to treat rust.'
        },
        {
            'question': 'How do I treat anthracnose?',
            'answer': 'Apply fungicides, avoid overhead watering, and remove affected plant parts to treat anthracnose.'
        },
    ]
}

def add_diseases(diseases):
    for disease_data in diseases:
        Disease.objects.create(
            name=disease_data['name'],
            symptoms=disease_data['symptoms'],
            description=disease_data['description'],
            prevention=disease_data['prevention'],
            treatment=disease_data['treatment'],
            image=disease_data['image']
        )
    print('Diseases added successfully.')

def add_faq_items(faq_data):
    for group_name, items in faq_data.items():
        group, created = FAQGroup.objects.get_or_create(name=group_name)
        for item_data in items:
            FAQItem.objects.create(
                group=group,
                question=item_data['question'],
                answer=item_data['answer']
            )
    print('FAQ items added successfully.')

if __name__ == '__main__':
    add_diseases(diseases)
    add_faq_items(faq_data)
