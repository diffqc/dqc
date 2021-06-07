apps_details = [
    {
        "app": "Learning xc functional from experimental data",
        "repo": "https://github.com/mfkasim1/xcnn",  # leave blank if no repo available
        # leave blank if no paper available, strongly suggested to link to open-access paper
        "paper": "https://arxiv.org/abs/2102.04229",
    },
    {
        "app": "Basis optimization",
        "repo": "https://github.com/diffqc/dqc-apps/tree/main/01-basis-opt",
        "paper": "",
    },
    {
        "app": "Alchemical perturbation",
        "repo": "https://github.com/diffqc/dqc-apps/tree/main/04-alchemical-perturbation",
        "paper": "",
    },
]

repo_icons = {
    "github": "docs/data/readme_icons/github.svg",
}

paper_icon = "docs/data/readme_icons/paper.svg"

def get_repo_name(repo_link):
    # get the repository name
    for repo_name in repo_icons.keys():
        if repo_name in repo_link:
            return repo_name
    raise RuntimeError("Unlisted repository, please contact admin to add the repository.")

def add_row(app_detail):
    # get the string for repository column
    if app_detail['repo'].strip() != "":
        repo_name = get_repo_name(app_detail['repo'])
        repo_detail = f"[![{repo_name}]({repo_icons[repo_name]})]({app_detail['repo']})"
    else:
        repo_detail = ""

    # get the string for the paper column
    if app_detail['paper'].strip() != "":
        paper_detail = f"[![Paper]({paper_icon})]({app_detail['paper']})"
    else:
        paper_detail = ""

    s = f"| {app_detail['app']} | {repo_detail} | {paper_detail} |\n"
    return s

def main():
    # construct the strings
    s = "| Applications                      | Repo | Paper |\n"
    s += "|-----------------------------------|------|-------|\n"
    for app_detail in apps_details:
        s += add_row(app_detail)

    # open the readme file
    fname = "README.md"
    with open(fname, "r") as f:
        content = f.read()

    # find the signature in README
    sig_start = "<!-- start of readme_appgen.py -->"
    sig_end = "<!-- end of readme_appgen.py -->"
    note = "<!-- Please do not edit this part directly, instead add your " + \
           "application in the readme_appgen.py file -->\n"
    idx_start = content.find(sig_start)
    idx_end = content.find(sig_end)

    # write the string to the README
    content = content[:idx_start] + sig_start + "\n" + note + s + content[idx_end:]
    with open(fname, "w") as f:
        f.write(content)

if __name__ == "__main__":
    main()
