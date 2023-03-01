from qiskit import IBMQ
from qiskit.providers.ibmq import IBMQJobManager
from dataclasses import asdict
from experiment_configuration import Qiskit, Functions

"""
This skript is used to retrieve jobs from IBMQ.
"""

small_job_set_ids = [
    # add the ids of the jobs here
]
large_job_set_ids = [
    # add the ids of the jobs here
]
job_set_ids = small_job_set_ids + large_job_set_ids


def retrieve():
    """
    Retrieve all the experiments from IBMQ and write it to a file.
    """

    provider = IBMQ.load_account()
    job_manager = IBMQJobManager()

    experiment_results = []

    for i, job_set_id in enumerate(job_set_ids):
        job_set = job_manager.retrieve_job_set(job_set_id=job_set_id, provider=provider)
        jobs = job_set.managed_jobs()
        print(f"Retrieve job set {i+1}/{len(job_set_ids)}: {job_set_id}")

        for job_collection in jobs:
            job_results = job_collection.result()
            for result in job_results.results:
                experiment_result = Qiskit.result_to_experiment_result(result)
                experiment_results.append(asdict(experiment_result))

    Functions.write_experiment_to_file(experiment_results, "ibm")

    return


if __name__ == "__main__":
    retrieve()
