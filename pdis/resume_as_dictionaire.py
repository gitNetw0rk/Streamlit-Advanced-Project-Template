import streamlit as st


def main():
    st.title("Resume as dictionaire")

    # Question
    st.info('Can you see the key-value pair "name": "Chuck Norris" in the provided code?')

    # Define manually dict
    chuck_norris_resume = {
        "name": "Chuck Norris",
        "title": "Legendary Martial Artist and Actor",
        "skills": [
            "Martial Arts Mastery",
            "Unparalleled Strength",
            "Indestructibility",
            "Intimidation",
            "Flawless Beard Growth",
        ],
        "experience": [
            {
                "Title": "World Champion Martial Artist",
                "Duration": "1970s-1980s",
                "Description": "Won numerous martial arts championships and became a dominant figure in the world of martial arts.",
            },
            {
                "Title": "Action Movie Star",
                "Duration": "1980s-Present",
                "Description": "Starred in numerous action films, becoming a cultural icon known for his tough, no-nonsense persona.",
            },
            {
                "Title": "Internet Meme",
                "Duration": "2000s-Present",
                "Description": "Achieved legendary status as an internet meme, with countless jokes and memes highlighting his superhuman abilities and toughness.",
            },
        ],
        "education": {
            "Degree": "Master of Roundhouse Kicks",
            "Institution": "The School of Hard Knocks",
        },
        "languages": ["English", "Kickass"],
        "hobbies": ["Roundhouse Kicking", "Beard Grooming", "Saving the World in his Spare Time"],
    }

    # Print it
    st.write(chuck_norris_resume)


if __name__ == "__main__":
    main()
