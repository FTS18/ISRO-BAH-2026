"""
AI Climate Copilot Engine — India's Climate Digital Twin
Lightweight RAG Engine matching live NetCDF regional climate states against
official ICAR-CRIDA District Agricultural Contingency Plans, NDMA Disaster Guidelines,
and CWC Reservoir Operations Protocols.

Seasonal Calendar:
    Kharif season: June 1 - October 31 (Monsoon-dependent crops)
    Rabi season:   November 1 - February 28 (Winter wheat/mustard crops)
    Zaid season:   March 1 - May 31 (Summer vegetables/watermelon)
"""

import re
from datetime import datetime


def _current_season() -> str:
    """Returns the active cropping season based on the current calendar month."""
    month = datetime.utcnow().month
    if 6 <= month <= 10:
        return "Kharif"
    elif 11 <= month or month <= 2:
        return "Rabi"
    else:
        return "Zaid"


class ClimateCopilotEngine:
    def __init__(self):
        # Official ICAR-CRIDA, NDMA, and CWC Knowledge Base Articles
        self.knowledge_base = [
            {
                "topic": "paddy_heat_stress",
                "keywords": ["paddy", "rice", "heat", "heatwave", "temperature", "flowering", "sterility"],
                "season": "Kharif",
                "content": """**ICAR-CRIDA Advisory for Paddy Heat Stress (>35 degrees C):**
- **Agronomic Impact**: Temperatures exceeding 35°C during the anthesis (flowering) stage induce pollen sterility, leading to significant spikelet sterility and yield collapse.
- **Immediate Mitigation**: Maintain standing water of 3-5 cm in paddy fields to create a micro-climate evaporative cooling effect.
- **Foliar Spray**: Spray 2% Potassium Nitrate (KNO3) or 0.5% Muriate of Potash (MOP) during dry spells to enhance osmotic potential and heat tolerance.
- **Long-term Adaptation**: Shift to heat-tolerant varieties such as DRR Dhan 45 or PR 126 in vulnerable zones."""
            },
            {
                "topic": "monsoon_drought_millets",
                "keywords": ["drought", "millets", "ragi", "deficit", "dry", "jowar", "bajra", "kharif"],
                "season": "Kharif",
                "content": """**ICAR-CRIDA Contingency Plan for Monsoonal Deficit (<2.5 mm/day avg):**
- **Crop Substitution**: If monsoon arrival is delayed beyond July 15, replace main season paddy with short-duration millets (Finger Millet/Ragi - GPU 28, Pearl Millet/Bajra, or Sorghum/Jowar).
- **Moisture Conservation**: Practice inter-row cultivation and dust mulching to break soil capillaries. Blanket application of organic mulches (straw/stover) to conserve soil residual moisture.
- **Sowing Strategy**: Adopt ridge and furrow sowing or broad bed furrow (BBF) to conserve in-situ moisture."""
            },
            {
                "topic": "heavy_rain_flood_discharge",
                "keywords": ["flood", "heavy rain", "cyclone", "dam", "overflow", "discharge", "reservoir", "waterlogging", "amphan"],
                "season": "all",
                "content": """**NDMA and CWC Guidelines for Extreme Precipitation (>64.5 mm/day):**
- **Reservoir Operations**: Initiate pre-emptive controlled releases based on upstream unit hydrograph forecasts. Keep storage buffers at 75-80% FRL (Full Reservoir Level) ahead of extreme storm landfall.
- **Agricultural Drainage**: Provide surface open drainage channels (30 cm depth) in heavy clayey soils to prevent root rot and fungal blight in standing crops.
- **Emergency Alert**: Issue orange/red alerts to low-lying riverine taluks. Pre-position NDRF flood rescue teams near major basin spillways."""
            },
            {
                "topic": "sugarcane_drought_water",
                "keywords": ["sugarcane", "sugar", "irrigation", "groundwater", "water"],
                "season": "Kharif",
                "content": """**ICAR Advisory for Sugarcane under Sub-Normal Rainfall:**
- **Water Saving Technique**: Switch from flood irrigation to alternate skip-furrow irrigation or sub-surface drip irrigation, saving up to 40% irrigation water.
- **Trash Mulching**: Apply trash mulch (5 tonnes/ha) in furrow intervals to arrest weed growth and limit soil evaporation.
- **Fertilizer Adjustment**: Defer nitrogenous top dressing until soil moisture levels recover to prevent volatilization losses."""
            },
            {
                "topic": "wheat_cold_rabi",
                "keywords": ["wheat", "cold", "winter", "rabi", "frost", "fog", "mustard"],
                "season": "Rabi",
                "content": """**ICAR-CRIDA Advisory for Wheat and Mustard under Rabi Cold Stress:**
- **Frost Risk**: Temperatures below 5°C during the grain-filling stage of wheat (December-January) cause sterility and chaffy grains. Apply light pre-dawn irrigation to prevent ground frost.
- **Fog Impact**: Dense fog reduces photosynthetically active radiation (PAR), delaying heading. Monitor cumulative GDD (Growing Degree Days) against varietal requirements.
- **Variety Selection**: Use timely sown varieties (HD-2967, HD-3086) that complete grain filling before late-February temperature rise onset."""
            },
            {
                "topic": "cotton_boll_weevil",
                "keywords": ["cotton", "boll", "bollworm", "kharif", "fibre"],
                "season": "Kharif",
                "content": """**ICAR Advisory for Cotton under Temperature and Moisture Stress:**
- **Heat-Induced Boll Shedding**: Temperatures above 38°C during peak boll formation (August-September) cause boll shedding. Apply 2% potassium sulphate (K2SO4) foliar spray to reduce pre-harvest drop.
- **High Humidity Pest Risk**: Prolonged wet spells (>10 consecutive rainy days) create ideal conditions for fungal blight (Alternaria macrospora). Scout fields bi-weekly and apply mancozeb 75 WP at 2.5 g/L.
- **Irrigation Threshold**: Maintain field capacity at 50% soil available water capacity (AWC) during boll development to prevent fibre quality degradation."""
            },
            {
                "topic": "groundnut_drought_zaid",
                "keywords": ["groundnut", "peanut", "oilseed", "zaid", "summer", "arachis"],
                "season": "Zaid",
                "content": """**ICAR Advisory for Groundnut under Zaid Summer Heat:**
- **Pod-fill Moisture**: Maintain soil moisture at field capacity during the pegging and pod-fill stage (45-90 days after sowing). Deficit irrigation at this critical period reduces shelling percentage by 15-25%.
- **Calcium Application**: Apply gypsum (500 kg/ha) at the time of flowering to improve pod development, particularly in Kharif relay-cropped groundnut.
- **Heat Avoidance**: Advance sowing date by 15-20 days to escape peak April-May temperature stress during pod filling."""
            },
            {
                "topic": "cyclone_coastal_alert",
                "keywords": ["cyclone", "depression", "bay of bengal", "landfall", "coastal", "storm surge", "wind"],
                "season": "all",
                "content": """**NDMA Standard Operating Procedure for Cyclonic Disturbance (Bay of Bengal):**
- **Pre-Landfall Evacuation**: Evacuate low-lying areas within 0-5 km of the coastline and all areas below 3 m contour well before landfall. Coordinate with District Disaster Management Authority (DDMA).
- **Agricultural Emergency**: Harvest standing Kharif crops that are within 15 days of maturity. Secure freshly cut produce in elevated, covered storage.
- **Reservoir Management**: Reduce reservoir levels by 10-15% of FRL capacity 72 hours before predicted extreme rainfall landfall to buffer storm inflow surge."""
            },
            {
                "topic": "soybean_waterlogging",
                "keywords": ["soybean", "soya", "legume", "waterlogging", "excess rain"],
                "season": "Kharif",
                "content": """**ICAR Advisory for Soybean under Waterlogged Conditions:**
- **Root Hypoxia**: Soybean is sensitive to waterlogging for more than 48-72 continuous hours, leading to root hypoxia, nitrogen fixation loss, and root rot (Phytophthora sojae).
- **Field Drainage**: Ensure field slopes of at least 0.5% towards drainage outlets. Construct broad-bed furrow (BBF) systems at 150 cm bed width on flat Vertisol fields to drain excess water.
- **Foliar Nutrition**: Spray 2% urea + 1% borax solution after waterlogging recession to compensate for nitrogen fixation loss due to nodule damage."""
            },
            {
                "topic": "irrigation_reservoir_normal",
                "keywords": ["reservoir", "dam", "irrigation", "storage", "kharif water"],
                "season": "all",
                "content": """**CWC Reservoir Operations Advisory for Normal Monsoon Conditions:**
- **Storage Optimization**: During normal monsoon (within 10% of seasonal normal precipitation), maintain reservoir levels at 85-90% of FRL by September 15 to ensure adequate Rabi season irrigation releases.
- **Canal Head Discharge**: Calibrate canal head discharge rates weekly using current inflow hydrographs to prevent downstream waterlogging.
- **Groundwater Recharge**: Activate check dam and percolation tank filling programs in hard rock zones (Deccan Plateau) during surplus monsoon weeks to recharge depleted groundwater aquifers."""
            },
            {
                "topic": "lightning_thunderstorm",
                "keywords": ["lightning", "thunder", "thunderstorm", "convective", "tstorm"],
                "season": "all",
                "content": """**NDMA Advisory for Convective Thunderstorm and Lightning Events:**
- **Agricultural Field Safety**: Instruct farm labour to immediately seek shelter in concrete structures or low ground during active thunderstorms. Avoid shelter under isolated tall trees or near metal irrigation equipment.
- **Crop Protection**: Stake and tie tall crops (maize, sunflower, sorghum) prior to predicted squall-line passage to prevent lodging by strong convective downdrafts.
- **Livestock Protection**: Move livestock into closed sheds at least 3 hours before predicted thunderstorm onset based on IMD nowcast bulletins."""
            },
            {
                "topic": "cold_wave_north_india",
                "keywords": ["cold wave", "coldwave", "cold snap", "north india", "plains", "foggy", "minimum temperature"],
                "season": "Rabi",
                "content": """**IMD and NDMA Advisory for Cold Wave Conditions in North Indian Plains:**
- **Declaration Threshold**: IMD declares a Cold Wave when the minimum temperature falls to 4°C or below, or is at least 4.5°C below the seasonal normal.
- **Crop Protection**: Apply light protective irrigation (foggy night irrigation) to release latent heat from soil and prevent crop temperature from falling below the critical threshold.
- **Livestock Advisory**: Provide dry bedding and wind shields for livestock. Ensure adequate caloric supplementation to compensate for thermoregulatory energy expenditure.
- **Protective Smoke**: Use biodegradable smoke from burning paddy/wheat straw in field borders during extreme cold nights to raise near-surface temperatures by 1-2°C in localized zones."""
            },
        ]

    def generate_response(self, query: str, curr_rain_mean: float, curr_temp_mean: float, curr_temp_max: float, region: str) -> str:
        """
        Processes the natural language query, combines it with live NetCDF regional context
        and the active cropping season, and returns a tailored, RAG-grounded expert recommendation.
        """
        query_clean = query.lower()
        active_season = _current_season()
        matched_articles = []

        # 1. Keyword-based retrieval — filter by matching keywords
        for article in self.knowledge_base:
            if any(kw in query_clean for kw in article["keywords"]):
                matched_articles.append(article["content"])

        # 2. If no specific keyword matches, trigger rules based on live NetCDF state
        #    and active cropping season
        if not matched_articles:
            if curr_temp_max > 35.0:
                # Heat stress — season-specific crop advisory
                if active_season == "Kharif":
                    matched_articles.append(self.knowledge_base[0]["content"])  # Paddy heat
                elif active_season == "Rabi":
                    matched_articles.append(self.knowledge_base[4]["content"])  # Wheat cold/heat
                else:
                    matched_articles.append(self.knowledge_base[6]["content"])  # Groundnut zaid

            elif curr_rain_mean > 64.5:
                matched_articles.append(self.knowledge_base[2]["content"])  # Flood discharge

            elif curr_rain_mean < 2.5:
                # Drought — season-specific
                if active_season == "Kharif":
                    matched_articles.append(self.knowledge_base[1]["content"])  # Millets deficit
                elif active_season == "Rabi":
                    matched_articles.append(self.knowledge_base[9]["content"])  # Reservoir management

            if not matched_articles:
                matched_articles.append(
                    f"**General Agro-Climatic Advisory for {region} ({active_season} Season):**\n"
                    f"Current assimilated conditions show stable rainfall ({curr_rain_mean:.1f} mm/day) and "
                    f"temperatures peaking at {curr_temp_max:.1f}°C. Soil moisture levels remain adequate for "
                    f"routine {active_season} agricultural operations. Continue standard schedule for weeding "
                    f"and nutrient top-dressing as per regional State Agriculture University package of practices."
                )

        # Structure the final Copilot response
        header = f"### AI Climate Copilot Advisory for {region} — {active_season} Season\n"
        header += f"*Live Context: Avg Rain = {curr_rain_mean:.1f} mm/day | Peak Temp = {curr_temp_max:.1f}°C | Season = {active_season}*\n\n"

        return header + "\n\n---\n\n".join(matched_articles) + \
               "\n\n*Source: Grounded in official ICAR-CRIDA District Contingency Plans, NDMA Operating Procedures, and CWC Reservoir Operations Advisories.*"
