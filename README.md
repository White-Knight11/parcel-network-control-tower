# Parcel Network Control Tower

A Streamlit-based control-tower demo for a parcel network that optimizes one-day shipping, compares sourcing options, and highlights SLA risk across a national network. The app focuses on a supply-chain control-room workflow for parcel operations, with New Jersey/Jersey City context embedded in the planning layer.

## Main app file

The project entrypoint is now `control_tower.py`.

## Live demo

[Open the Parcel Network Control Tower](https://parcel-network-control-tower.streamlit.app/)

## Run locally

```bash
cd "/Users/sunidhirusia/Documents/Supply Chain Project"
python3 -m pip install -r requirements.txt
python3 -m streamlit run control_tower.py
```

You can also use the compatibility entrypoint:

```bash
python3 -m streamlit run app.py
```

## GitHub / deployment notes

This project is GitHub-ready as a standard Python app. You still need to create or connect a GitHub repository and push it from your own account.

For a public Streamlit deployment, the easiest path is to:

1. create a GitHub repo
2. push the project
3. deploy from Streamlit Community Cloud using the repo and branch
4. set the main file to `control_tower.py`

## Notes

This dashboard uses synthetic planning data for demonstration and is not connected to live carrier or warehouse systems.
