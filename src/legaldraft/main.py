#!/usr/bin/env python
import sys
import warnings

from legaldraft.crew import Legaldraft

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': '''On the morning of 12th June 2023, I, Priya Sharma, confirm that I am the lawful owner of the residential property located at 72, Green Avenue, Lajpat Nagar, New Delhi, 110024. This property was purchased by me on 25th March 2019, as per the sale deed registered with the New Delhi Sub-Registrar Office, bearing document number 224/2019. I declare that there are no existing mortgages, liens, or any other encumbrances on the property. The property has remained in my sole possession since the date of purchase, and no other party has claimed ownership or legal rights to it.
        I am willing to provide copies of all relevant ownership documents and am prepared to testify to this information under oath, if required. Create an affidavit for this in delhi high court'''
    }
    Legaldraft().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'topic': '''On the morning of 12th June 2023, I, Priya Sharma, confirm that I am the lawful owner of the residential property located at 72, Green Avenue, Lajpat Nagar, New Delhi, 110024. This property was purchased by me on 25th March 2019, as per the sale deed registered with the New Delhi Sub-Registrar Office, bearing document number 224/2019. I declare that there are no existing mortgages, liens, or any other encumbrances on the property. The property has remained in my sole possession since the date of purchase, and no other party has claimed ownership or legal rights to it.I am willing to provide copies of all relevant ownership documents and am prepared to testify to this information under oath, if required. Create an affidavit for this in delhi high court'''
    }
    try:
        Legaldraft().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Legaldraft().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'topic': '''On the morning of 12th June 2023, I, Priya Sharma, confirm that I am the lawful owner of the residential property located at 72, Green Avenue, Lajpat Nagar, New Delhi, 110024. This property was purchased by me on 25th March 2019, as per the sale deed registered with the New Delhi Sub-Registrar Office, bearing document number 224/2019. I declare that there are no existing mortgages, liens, or any other encumbrances on the property. The property has remained in my sole possession since the date of purchase, and no other party has claimed ownership or legal rights to it. 
        I am willing to provide copies of all relevant ownership documents and am prepared to testify to this information under oath, if required. Create an affidavit for this in delhi high court'''
    }
    try:
        Legaldraft().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
