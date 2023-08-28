import argparse
import os
import sys
import os.path as osp
import pickle

import yaml
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def save_object(obj, filename):
    with open(filename, 'wb') as out_file:  # Overwrites any existing file.
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp_file:  # Overwrites any existing file.
        out = pickle.load(inp_file)
    return out


class SlurmJobManager:
    cache_dir = "~/.jobrunner"
    cache_file_status = "jobrunner_status_table.csv"

    def __init__(self, file, overwrite=False):
        self.cache_dir = self.__class__.cache_dir
        self.cache_file_status = self.__class__.cache_file_status
        self.file = file
        self.overwrite = overwrite

        self.jobs = self.build_jobs()

        if osp.exists(osp.join(osp.expanduser(self.cache_dir), self.cache_file_status)):
            self.status_table = pd.read_csv(osp.join(osp.expanduser(self.cache_dir), self.cache_file_status))

        else:
            self.status_table = pd.DataFrame({
                "job_id": [], "group_name": [], "job_status": [], "preamble": [], "command": []
            })

        if not osp.exists(osp.expanduser(self.cache_dir)):
            os.makedirs(osp.expanduser(self.cache_dir))

    @staticmethod
    def parse_file(filepath):
        with open(filepath, "r") as stream:
            return yaml.safe_load(stream)

    def build_jobs(self):

        parsed = self.parse_file(self.file)
        jobs = []

        common_decl = "\n".join(parsed["common_preamble_declarations"])
        common_runs = "\n".join(parsed["common_preamble_runs"])

        for group in parsed["groups"]:
            group_preamble = "\n".join(group["preamble"])
            for job in group['paralleljobs']:
                jobs.append({
                    "group_name": group["name"],
                    "preamble": "{}\n{}\n{}".format(common_decl, group_preamble, common_runs),
                    "command": "{}".format(job)
                })

        return jobs

    def submit_jobs(self):

        if self.overwrite:
            pd.set_option('display.max_colwidth', 150)
            print(self.status_table[['job_id', 'job_status', 'command']])
            overwrite_string = input("Overwrite previous jobs? Enter job(s) to overwrite, or 'n' to skip: ")
            overwrite_commands = [
                self.status_table.loc[int(s)]['command'] for s in overwrite_string.split(",")
            ]
        else:
            overwrite_commands = []

        for job in self.jobs:

            #  If job is already running (successfully), skip
            if (job["command"] in self.status_table["command"].values and
                    not job["command"] in overwrite_commands and
                    not self.status_table[self.status_table["command"] == job["command"]]["job_status"].values[
                        0].startswith("FAILED") and
                    not self.status_table[self.status_table["command"] == job["command"]]["job_status"].values[
                        0].startswith("CANCELLED")):

                print("Skipping job: {} (Already running)".format(job["group_name"]))

            else:

                # Drop previous row
                # if job["command"] in self.status_table["command"].values and job["command"] in overwrite_commands:
                #     self.status_table = self.status_table[self.status_table["command"] != job["command"]]

                print("Running job: {}".format(job["group_name"]))
                print(os.popen("rm tmp.sh").read())
                print(os.popen("touch tmp.sh").read())
                print(os.popen("echo '{}' >> tmp.sh".format(job["preamble"])).read())
                print(os.popen("echo '{}' >> tmp.sh".format(job["command"])).read())

                out = os.popen("sbatch tmp.sh").read()

                print("Running: \n{}".format(out))
                job_id = out.split(" ")[-1].strip()

                new_row = {'job_id': job_id, 'group_name': job["group_name"], 'job_status': 'SUBMIT',
                           'preamble': job['preamble'], 'command': job['command']}

                # Update previous row if exists
                if job["command"] in self.status_table["command"].values:
                    # import pdb; pdb.set_trace()
                    new_row["job_status"] = "{}*".format(new_row["job_status"])
                    # self.status_table.loc[self.status_table["command"] == job["command"]] = pd.DataFrame([new_row])
                    mask = self.status_table["command"] == job["command"]
                    for col, value in new_row.items():
                        self.status_table.loc[mask, col] = value

                else:
                    self.status_table = pd.concat([self.status_table, pd.DataFrame([new_row])], ignore_index=True)

        outpath = osp.join(osp.expanduser(self.cache_dir), self.cache_file_status)
        df = pd.DataFrame(self.status_table)
        df.to_csv(outpath, index=False)
        print(self.status_table[['job_id', 'job_status', 'command']])

    @classmethod
    def status(cls):
        outpath = osp.join(osp.expanduser(cls.cache_dir), cls.cache_file_status)
        status_table = pd.read_csv(outpath)

        status = []

        for job_id in status_table['job_id']:
            out = os.popen("sacct -j {} --format state".format(job_id)).read()
            status_i = out.split("\n")[2].strip()
            # If status in table ensds with *, then add * to status_i
            if status_table[status_table['job_id'] == job_id]['job_status'].values[0].endswith("*"):
                status_i = "{}*".format(status_i)
            status.append(status_i)
        status_table['job_status'] = status
        status_table.to_csv(outpath)

        pd.set_option('display.max_colwidth', 150)
        reduced = status_table[['job_id', 'job_status', 'command']]
        print(reduced)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, )
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    parser.add_argument('-s', '--status', action='store_true', default=False)
    args = parser.parse_args()

    if args.status:
        SlurmJobManager.status()
    else:
        SlurmJobManager(args.filepath, args.overwrite).submit_jobs()
