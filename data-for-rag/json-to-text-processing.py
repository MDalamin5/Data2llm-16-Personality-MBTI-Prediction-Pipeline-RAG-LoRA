import json
import os

def json_to_text(data):
    text = ''
    
    # Personal Info
    personal = data.get('personal_info', {})
    text += f"Name: {personal.get('name', 'N/A')}\n"
    text += f"Headline: {personal.get('headline', 'N/A')}\n"
    text += f"Location: {personal.get('location', 'N/A')}\n"
    text += f"Current Company: {personal.get('current_company', 'N/A')}\n\n"
    
    # About
    text += f"About: {data.get('about', 'N/A')}\n\n"
    
    # Top Skills
    skills = data.get('top_skills', [])
    text += "Top Skills: " + ', '.join(skills) + "\n\n"
    
    # Featured Section
    featured = data.get('featured_section', [])
    text += "Featured Section:\n"
    for item in featured:
        title = item.get('title') or item.get('type', 'N/A')
        desc = item.get('description', '') or ''
        link = item.get('link', '')
        text += f"- {title}: {desc} (Link: {link})\n"
    text += "\n"
    
    # Experience
    experience = data.get('experience', [])
    text += "Experience:\n"
    for exp in experience:
        title = exp.get('title', 'N/A')
        company = exp.get('company', 'N/A')
        emp_type = exp.get('employment_type', 'N/A')
        duration = exp.get('duration', 'N/A')
        location = exp.get('location', 'N/A')
        desc = exp.get('description', '') or 'N/A'
        text += f"- {title} at {company} ({emp_type}, {duration}, {location}): {desc}\n"
    text += "\n"
    
    # Education
    education = data.get('education', [])
    text += "Education:\n"
    for edu in education:
        school = edu.get('school', 'N/A')
        degree = edu.get('degree', 'N/A')
        duration = edu.get('duration', 'N/A')
        grade = edu.get('grade', '') or 'N/A'
        text += f"- {degree} from {school} ({duration}, Grade: {grade})\n"
    text += "\n"
    
    # Projects
    projects = data.get('projects', [])
    text += "Projects:\n"
    for proj in projects:
        title = proj.get('title', 'N/A')
        duration = proj.get('duration', 'N/A')
        desc = proj.get('description', '') or 'N/A'
        text += f"- {title} ({duration}): {desc}\n"
    text += "\n"
    
    # Interests
    interests = data.get('interests', [])
    text += "Interests:\n"
    for interest in interests:
        name = interest.get('name', 'N/A')
        followers = interest.get('followers', 'N/A')
        text += f"- {name} ({followers})\n"
    text += "\n"
    
    # Posts (treat all equally, list them)
    posts = data.get('posts', [])
    text += "Posts:\n"
    for i, post in enumerate(posts, 1):
        content = post.get('content', 'N/A')
        images = ', '.join(post.get('image_links', [])) or 'None'
        text += f"Post {i}: {content}\nImages: {images}\n\n"
    
    return text

# Directory with JSON files
input_dir = "persons-data-Linkedin-json/"
output_dir = "processed/"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        text = json_to_text(data)
        
        output_filename = filename.replace('.json', '.txt')
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            f.write(text)
        
        print(f"Processed {filename} to {output_filename}")