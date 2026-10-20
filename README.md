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

## Deployment

The public GitHub repository is [White-Knight11/parcel-network-control-tower](https://github.com/White-Knight11/parcel-network-control-tower).
The public Streamlit deployment is linked above and deploys `control_tower.py`
from the `main` branch.

## Notes

This dashboard uses synthetic planning data for demonstration and is not connected to live carrier or warehouse systems.
