# import os
# import sys
from typing import Any, Callable
from .CONSTANTS import (
    KEY_ALL_JOBEST_IDS,
    KEY_RQ_WORKER_JOBS,
)
from .dao import Dao
from .small_memory import SmallMemory


def report_failure(job: Job, connection: Any, type: Any, value: Any, traceback: Any):
    print(job, connection, type, value, traceback)


# NOTE: this is used both in fastAPI backend (/captain) and Django backend (/server)
# however, most of these methods are redundant in fastAPI backend.
# TODO: if we completely get rid of old backend, figure out which methods are needed
class JobService:
    def __init__(self, maximum_runtime: float = 3000):
        self.dao = Dao.get_instance()

    def get_all_jobs(self):
        all_jobs = self.dao.get_obj(KEY_RQ_WORKER_JOBS)
        return all_jobs

    def delete_all_rq_worker_jobs(self, nodes: list[str]):
        for node_id in nodes:
            try:
                job = Job.fetch(node_id, connection=self.dao.r)
            except NoSuchJobError:
                continue
            except Exception:
                print(" Failed to cancel job: ", node_id + ", ignoring..")
                continue
            if job:
                print("Deleting job: ", job.id)
                job.delete()
        self.dao.delete_object(KEY_RQ_WORKER_JOBS)

    def delete_all_jobset_data(self):
        all_running_jobest_ids = self.dao.get_set_list(KEY_ALL_JOBEST_IDS)
        for jobset_id in all_running_jobest_ids:
            jobset_node_keys = f"{jobset_id}_ALL_NODES"
            self.dao.delete_object(jobset_node_keys)

    def add_jobset_id(self, jobset_id: str):
        self.dao.add_to_set(KEY_ALL_JOBEST_IDS, jobset_id)

    def add_to_redis_obj(self, key: str, value: dict[str, Any]):
        self.dao.set_obj(key, value)

    def get_jobset_data(self, key: str):
        return self.dao.get_obj(key)

    def enqueue_job(
        self,
        func: Any,
        jobset_id: str,
        job_id: str,
        iteration_id: str,
        ctrls: dict[str, Any],
        previous_jobs: list[Any],
    ):
        if Job.exists(job_id, self.dao.r):
            job = Job.fetch(job_id, connection=self.dao.r)
            job.delete()

        job = self.queue.enqueue(
            func,
            job_timeout="15m",
            on_failure=report_failure,
            job_id=iteration_id,
            kwargs={
                "ctrls": ctrls,
                "previous_jobs": previous_jobs,
                "jobset_id": jobset_id,
                "node_id": job_id,
                "job_id": iteration_id,
            },
            depends_on=[],
        )
        self.add_job(iteration_id, jobset_id)

        return job

    def add_job(self, job_id: str, jobset_id: str):
        self.dao.add_to_list(f"{jobset_id}_ALL_NODES", job_id)

    def fetch_job(self, job_id: str):
        try:
            return Job.fetch(job_id, connection=self.dao.r)
        except Exception:
            print("Error in job fetching for job id: ", job_id)
            return None

    def reset(self, nodes: list[Any]):
        self.delete_all_rq_worker_jobs(nodes)
        self.delete_all_jobset_data()
        SmallMemory().clear_memory()
