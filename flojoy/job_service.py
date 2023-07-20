from typing import Any
from .dao import Dao


"""
Service that allows to manage jobs
"""


class JobService:
    def get_job_result(self, job_id: str | None) -> dict[str, Any] | None:
        if job_id is None:
            return None
        return Dao.get_instance().get_job_result(job_id)

    def post_job_result(self, job_id: str, result: Any):
        Dao.get_instance().post_job_result(job_id, result)

    def job_exists(self, job_id: str) -> bool:
        return Dao.get_instance().job_exists(job_id)

    def delete_job(self, job_id: str):
        Dao.get_instance().delete_job(job_id)

    def reset(self):
        Dao.get_instance().clear_job_results()
