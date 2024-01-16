import os
from typing import Dict
import git
import subprocess
import platform
import numpy as np
import datetime


def interp_2d(x, xp, fp):
    n_f = fp.shape[0]
    f = np.zeros((n_f, x.shape[0]))
    for i in range(n_f):
        f[i, :] = np.interp(x, xp, fp[i, :])
    return f


def get_2d_ndarray(v: np.ndarray) -> np.ndarray:
    if len(v.shape) == 1:
        x = np.reshape(v, (1, v.size))
    else:
        x = v
    return x


def get_repository_tree(relative_path: bool = False) -> Dict:
    repo_tree = {}
    if relative_path:
        repo_tree["repository"] = ""
    else:
        repo_tree["repository"] = git.Repo(".", search_parent_directories=True).working_tree_dir
    repo_tree["src"] = os.path.join(repo_tree["repository"], "src")
    repo_tree["output"] = os.path.join(repo_tree["repository"], "aerodynamics", "output")
    repo_tree["planes"] = os.path.join(repo_tree["repository"], "aerodynamics", "planes")
    repo_tree["3d_analysis"] = os.path.join(repo_tree["repository"], "aerodynamics", "analyses_3d")
    repo_tree["polars"] = os.path.join(repo_tree["repository"], "aerodynamics", "polars")
    repo_tree["scripts"] = os.path.join(repo_tree["repository"], "aerodynamics", "scripts")
    repo_tree["resources"] = os.path.join(repo_tree["src"], "aero", "resources")
    repo_tree["database_aerodynamic_models"] = os.path.join("aerodynamics", "database_aerodynamic_models")
    repo_tree["ros_launch"] = os.path.join(repo_tree["src"], "ros_muav", "launch")
    repo_tree["urdf"] = os.path.join(repo_tree["src"], "ros_muav", "urdf")
    repo_tree["database_propeller"] = os.path.join(repo_tree["src"], "cots", "propeller")
    repo_tree["database_servomotor"] = os.path.join(repo_tree["src"], "cots", "servomotor")
    repo_tree["pickle_codesign_tasks"] = os.path.join(repo_tree["src"], "pickle_codesign_tasks")
    return repo_tree


def get_date_str():
    now = datetime.datetime.now()
    year = "{:02d}".format(now.year)
    month = "{:02d}".format(now.month)
    day = "{:02d}".format(now.day)
    hour = "{:02d}".format(now.hour)
    minute = "{:02d}".format(now.minute)
    second = "{:02d}".format(now.second)
    return f"{year}-{month}-{day}_{hour}h{minute}m{second}s"


class Markdown:
    @staticmethod
    def create_summary(title, str):
        out = "<details>\n<summary><b>%s</b></summary>\n\n%s\n\n</details>\n" % (title, str)
        return out


class Project_status:
    def __init__(self) -> None:
        pc = self.get_pc_info()
        pip = self.get_pip_status()
        repo = self.get_current_repo_status()
        self.out = ""
        self.out += repo["commit"] + "\n"
        self.out += Markdown.create_summary("full status", repo["recap"] + pip["recap"] + pc["recap"])

    def create_report(self, name_file: str) -> None:
        with open(f"{name_file}.md", "w") as f:
            f.write(self.out)
        print(f"{name_file}.md created")

    @staticmethod
    def get_current_repo_status():
        repo_tree = get_repository_tree()
        name_repo = repo_tree["repository"].split("/")[-1]
        out = {}
        out["status"] = "``` bash\n" + subprocess.getoutput("git status -s") + "\n```"
        out["diff"] = "``` diff\n" + subprocess.getoutput("git diff") + "\n```"
        out["hash"] = subprocess.getoutput("git rev-parse HEAD")
        out["name_repo"] = name_repo
        out["url"] = (
            git.Repo(".", search_parent_directories=True)
            .remotes.origin.url.replace("git@github.com:", "https://github.com/")
            .replace(".git", "")
        )
        out["commit"] = "`%s`: %s/commit/%s" % (out["name_repo"], out["url"], out["hash"])
        str = ""
        for f in ["commit", "status", "diff"]:
            str += "\n**%s**\n %s \n" % (f, out[f])
        out["recap"] = "## " + out["name_repo"] + "\n" + str + "\n"
        return out

    @staticmethod
    def get_pip_status():
        out = {}
        out["list"] = "``` bash\n" + subprocess.getoutput("pip list") + "\n```"
        out["recap"] = "## pip\n #### list\n" + out["list"] + "\n"
        return out

    @staticmethod
    def get_pc_info():
        out = {}
        out["name"] = platform.node()
        out["vers"] = platform.platform()
        out["recap"] = "## laptop\n" + "name: `" + out["name"] + "`\n" + "vers: `" + out["vers"] + "`\n"
        return out


if __name__ == "__main__":
    print(get_repository_tree())
