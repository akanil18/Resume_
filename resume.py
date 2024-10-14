import streamlit as st
import requests
import json
import pdfplumber
import os
from bs4 import BeautifulSoup
import re
import string
import nltk
from nltk.corpus import stopwords
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional

nltk.download('stopwords')

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_stop_words(text)
    return text

from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional

class PersonalInformation(BaseModel):
    name: str
    email: str
    mobile: str
    city: str
    country: str
    linkedIn: Optional[HttpUrl]
    gitHub: Optional[HttpUrl]

    class Config:
        arbitrary_types_allowed = True  

class Skills(BaseModel):
    languages: dict
    frameworks: dict
    technologies: dict
    total_skill_experience: dict
    llm_experience: bool
    gen_ai_experience: bool

    class Config:
        arbitrary_types_allowed = True 

class Education(BaseModel):
    school_name: str
    degree_name: str
    city: str
    country: str
    year_of_start: int
    year_of_graduation: int
    duration_in_years: float
    mode: str
    degree_level: str
    is_cs_degree: bool
    is_ml_degree: bool
    institute_type: str

    @field_validator('year_of_start', 'year_of_graduation', mode='before')
    def check_year(cls, v):
        if v < 1900:
            raise ValueError('Year must be greater than or equal to 1900')
        return v

    class Config:
        arbitrary_types_allowed = True  

class Project(BaseModel):
    project_name: str
    project_description: str

    class Config:
        arbitrary_types_allowed = True  

class CompanyInformation(BaseModel):
    name: str
    last_position_held: str
    city: str
    country: str
    joining_month_and_year: str
    leaving_month_and_year: str
    total_duration_in_years: float
    company_size_range: str
    total_capital_raised: str
    company_type: str
    is_faang: bool
    has_the_company_raised_capital_in_last_5_years: Optional[bool]
    is_startup: Optional[bool]

    class Config:
        arbitrary_types_allowed = True  

class Experience(BaseModel):
    company_information: CompanyInformation
    positions_held_within_the_company: List[dict]

    class Config:
        arbitrary_types_allowed = True  

class ResumeData(BaseModel):
    personal_information: PersonalInformation
    title: str
    skills: Skills
    education: Education
    experience: List[Experience]
    projects_outside_of_work: List[Project]
    additional_experience_summary: dict
    achievements_awards: dict
    overall_summary_of_candidate: str

    class Config:
        arbitrary_types_allowed = True  



expected_json_format = {
    "personal_information": {
        "name": "<full name>",
        "email": "<email>",
        "mobile": "<mobile number>",
        "city": "<City if not present then current work place >",
        "country": "<Country>",
        "linkedIn": "<LinkedIn URL or null>",
        "gitHub": "<GitHub URL or null>"
    },
    "title": "<Type_of_Engineer> with <total_years_of_experience>",
    "skills": {
        "languages": {
            "proficient": ["<List the first 3-4 languages>"],
            "average": ["<List remaining languages>"]
        },
        "frameworks": {
            "proficient": ["<List the first 2-3 frameworks>"],
            "average": ["<List remaining frameworks>"]
        },
        "technologies": {
            "proficient": ["<List the first 2-3 technologies>"],
            "average": ["<List remaining technologies>"]
        },
        "total_skill_experience": {
            "<skill_name>": "<Experience in Years>"
        },
        "llm_experience": False,
        "gen_ai_experience": False
    },
    "education": {
        "school_name": "<School Name>",
        "degree_name": "<Degree Name>",
        "city": "<City>",
        "country": "<Country>",
        "year_of_start": "<Start Year>",
        "year_of_graduation": "<Graduation Year>",
        "duration_in_years": "<Duration in Years>",
        "mode": "<offline/online>",
        "degree_level": "<bachelors/masters/PhD>",
        "is_cs_degree": False,
        "is_ml_degree": False,
        "institute_type": "<Type of Institute>"
    },
    "experience": [
        {
            "company_information": {
                "name": "<Company Name>",
                "last_position_held": "<Last Position Held>",
                "City": "<City>",
                "country": "<Country>",
                "joining_month_and_year": "<MM-YYYY>",
                "leaving_month_and_year": "<MM-YYYY>",
                "total_duration_in_years": "<Total Duration in Years>",
                "company_size_range": "<Company Size Range>",
                "total_capital_raised": "<Total Capital Raised>",
                "company_type": "<Product/Service>",
                "is_faang": False,
                "has_the_company_raised_capital_in_the_last_5_years?": "<Yes/No/Unknown>",
                "is_startup": "<True/False/Unknown>"
            },
            "positions_held_within_the_company": [
                {
                    "position_name": "<Position Name>",
                    "position_starting_Date": "<MM-YYYY>",
                    "position_ending_date": "<MM-YYYY>",
                    "projects": [
                        {
                            "project_name": "<Project Name>",
                            "project_description": "<Project Description>"
                        }
                    ]
                }
            ]
        }
    ],
    "projects_outside_of_work": [
    {
      "project_name": "<Project Name>",
      "project_description": "<Project Description>"
    }
  ],
  "additional_experience_summary": {
    "years_of_full_time_experience_after_graduation": "<Years>",
    "totalstartup_experience": "<Years>",
    "total_early_stage_startup_experience": "<Years>",
    "product_company_experience": "<Years>",
    "service_company_experience": "<Years>",
    "gen_ai_experience": False
  },
  "achievements_awards": {
    "summary_of_achievements_awards": [
      "<Summary of Achievements and Awards>"
    ],
    "position_blurbs": [
      {
        "position_name": "<Position Name>",
        "blurb": "<Blurb for Position>"
      }
    ]
  },
  "overall_summary_of_candidate": "<Brief Summary of the Candidate>"

}



def call_api(resume_text, api_key, api_url):
    prompt = (
        f"Extract the output information from the resume not in json :\n\n"
        f"{json.dumps(expected_json_format, indent=2)}\n\n"
        f"Provide the extracted information in JSON format."
    )

    payload = json.dumps({
        "messages": [
            {
                "role": "system",
                "content": prompt + "\n\n" + resume_text
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4000
    })

    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        response_json = response.json()
        try:
            validated_data = ResumeData(**response_json)
            st.json(validated_data.model_dump())
            return validated_data
        except Exception as e:
            st.error(f"Validation error: {str(e)}")
            st.json(response_json)  
            return None
    else:
        st.error(f"API call failed with status code {response.status_code}: {response.text}")
        return None

def process_resumes(folder_path, api_key, api_url):
    try:
        c = 0
        total_resumes = len([f for f in os.listdir(folder_path) if f.endswith('.pdf')])

        for resume_file in os.listdir(folder_path):
            if resume_file.endswith('.pdf'):
                c += 1
                resume_path = os.path.join(folder_path, resume_file)
                st.write(f"Processing Resume {c} of {total_resumes}: {resume_file}...")

                resume_text = extract_text_from_pdf(resume_path)
                resume_text = preprocess_text(resume_text)

             
                summary = call_api(resume_text, api_key, api_url)
               
                if summary:
                    st.success(f"Successfully processed: {resume_file}")
                else:
                    st.error(f"Error processing: {resume_file}")

                st.write("---")

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")

# Streamlit UI
def main():
    st.title("Resume Parsing with API and Validation")

    api_url = #your url
    api_key = #your api

    folder_path = st.text_input("Folder Path for PDF Resumes")

    if st.button("Process Resumes"):
        if api_key and api_url and folder_path:
            process_resumes(folder_path, api_key, api_url)
        

if __name__ == "__main__":
    main()
