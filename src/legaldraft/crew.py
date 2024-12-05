from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import os

from dotenv import load_dotenv

load_dotenv()

# Uncomment the following line to use an example of a custom tool
# from legaldraft.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool
from crewai_tools import SerperDevTool

search_tool = SerperDevTool(serper_api_key=os.getenv('SERPER_API_KEY'))
# top_p=1.0 -> Disable sampling diversity
Tier1_llm=LLM(model="ft:gpt-4o-mini-2024-07-18:synlex-technologies:synlex-drafting-mini6-5:ATIzJYf6",temperature=0.0,seed=42,max_tokens=None,stop_sequences=None)
Tier2_llm=LLM(model="ft:gpt-4o-mini-2024-07-18:synlex-technologies:synlex-draft-p2:AUBQj7t7",temperature=0.0,seed=42,max_tokens=None,stop_sequences=None)
Tier3_llm=LLM(model="ft:gpt-4o-mini-2024-07-18:synlex-technologies:synlex-affidavit-p4:AYRxLrDR",temperature=0.0,seed=42,max_tokens=None,stop_sequences=None)


# OPENAI_API_KEY="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"
# OPENAI_MODEL_NAME="ft:gpt-4o-mini-2024-07-18:synlex-technologies:synlex-drafting-mini6-5:ATIzJYf6"

@CrewBase
class Legaldraft():
	"""Legaldraft crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def Tier1_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['Tier1_agent'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			llm=Tier1_llm
		)

	@agent
	def Tier2_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['Tier2_agent'],
			verbose=True,
			llm=Tier2_llm
		)

	@agent
	def Tier3_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['Tier3_agent'],
			verbose=True,
			llm=Tier3_llm
			# tools=[search_tool]
		)

	@task
	def Tier1_task(self) -> Task:
		return Task(
			config=self.tasks_config['Tier1_task'],
		)

	@task
	def Tier2_task(self) -> Task:
		return Task(
			config=self.tasks_config['Tier2_task'],
            # output_file='report.md',
            # expected_output="""A comprehensive legal document that includes:
			# 	1. Thorough legal research findings from Indian law sources
			# 	2. Properly formatted court document following Indian standards
			# 	3. Relevant case laws and statutory references
			# 	4. Jurisdiction-specific compliance details
			# 	5. Complete supporting documentation"""   
		)
  
	@task
	def Tier3_task(self) -> Task:
		return Task(
			config=self.tasks_config['Tier3_task'],
			output_file='report.md',
            expected_output="""A comprehensive legal document that includes:
				1. Thorough legal research findings from Indian law sources
				2. Properly formatted court document following Indian standards
				3. Relevant case laws and statutory references
				4. Jurisdiction-specific compliance details
				5. Complete supporting documentation"""		
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Legaldraft crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
