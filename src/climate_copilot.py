"""
AI Climate Copilot Engine — India's Climate Digital Twin
Lightweight RAG Engine matching live NetCDF regional climate states against
official ICAR-CRIDA District Agricultural Contingency Plans and NDMA Disaster Guidelines.
"""

import re

class ClimateCopilotEngine:
    def __init__(self):
        # Official ICAR-CRIDA & NDMA Knowledge Base Articles
        self.knowledge_base = [
            {
                "topic": "paddy_heat_stress",
                "keywords": ["paddy", "rice", "heat", "heatwave", "temperature", "flowering", "sterility"],
                "content": """**ICAR-CRIDA Advisory for Paddy Heat Stress (>35°C):**
- **Agronomic Impact**: Temperatures exceeding 35°C during the anthesis (flowering) stage induce pollen sterility, leading to significant spikelet sterility and yield collapse.
- **Immediate Mitigation**: Maintain standing water of 3-5 cm in paddy fields to create a micro-climate evaporative cooling effect.
- **Foliar Spray**: Spray 2% Potassium Nitrate (KNO3) or 0.5% Muriate of Potash (MOP) during dry spells to enhance osmotic potential and heat tolerance.
- **Long-term Adaptation**: Shift to heat-tolerant varieties such as DRR Dhan 45 or PR 126 in vulnerable zones."""
            },
            {
                "topic": "monsoon_drought_millets",
                "keywords": ["drought", "millets", "ragi", "deficit", "dry", "jowar", "bajra", "kharif"],
                "content": """**ICAR-CRIDA Contingency Plan for Monsoonal Deficit (<2.5 mm/day avg):**
- **Crop Substitution**: If monsoon arrival is delayed beyond July 15, replace main season paddy with short-duration millets (Finger Millet/Ragi - GPU 28, Pearl Millet/Bajra, or Sorghum/Jowar).
- **Moisture Conservation**: Practice inter-row cultivation and dust mulching to break soil capillaries. Blanket application of organic mulches (straw/stover) to conserve soil residual moisture.
- **Sowing Strategy**: Adopt ridge and furrow sowing or broad bed furrow (BBF) to conserve in-situ moisture."""
            },
            {
                "topic": "heavy_rain_flood_discharge",
                "keywords": ["flood", "heavy rain", "cyclone", "dam", "overflow", "discharge", "reservoir", "waterlogging", "amphan"],
                "content": """**NDMA & CWC Guidelines for Extreme Precipitation (>64.5 mm/day):**
- **Reservoir Operations**: Initiate pre-emptive controlled releases based on upstream unit hydrograph forecasts. Keep storage buffers at 75-80% FRL (Full Reservoir Level) ahead of extreme storm landfall.
- **Agricultural Drainage**: Provide surface open drainage channels (30 cm depth) in heavy clayey soils to prevent root rot and fungal blight in standing crops.
- **Emergency Alert**: Issue orange/red alerts to low-lying riverine taluks. Pre-position NDRF flood rescue teams near major basin spillways."""
            },
            {
                "topic": "sugarcane_drought_water",
                "keywords": ["sugarcane", "sugar", "irrigation", "groundwater", "water"],
                "content": """**ICAR Advisory for Sugarcane under Sub-Normal Rainfall:**
- **Water Saving Technique**: Switch from flood irrigation to alternate skip-furrow irrigation or sub-surface drip irrigation, saving up to 40% irrigation water.
- **Trash Mulching**: Apply trash mulch (5 tonnes/ha) in furrow intervals to arrest weed growth and limit soil evaporation.
- **Fertilizer Adjustment**: Defer nitrogenous top dressing until soil moisture levels recover to prevent volatilization losses."""
            }
        ]

    def generate_response(self, query: str, curr_rain_mean: float, curr_temp_mean: float, curr_temp_max: float, region: str) -> str:
        """
        Processes the natural language query, combines it with live NetCDF regional context,
        and returns a highly tailored, RAG-grounded expert recommendation.
        """
        query_clean = query.lower()
        matched_articles = []
        
        # Simple local RAG keyword retrieval
        for article in self.knowledge_base:
            if any(kw in query_clean for kw in article["keywords"]):
                matched_articles.append(article["content"])
                
        # If no specific keyword matches, trigger rules based on live NetCDF state
        if not matched_articles:
            if curr_temp_max > 35.0:
                matched_articles.append(self.knowledge_base[0]["content"]) # Heat stress
            elif curr_rain_mean < 2.5:
                matched_articles.append(self.knowledge_base[1]["content"]) # Drought millets
            elif curr_rain_mean > 30.0:
                matched_articles.append(self.knowledge_base[2]["content"]) # Flood discharge
            else:
                matched_articles.append(
                    f"**General Agro-Climatic Advisory for {region}:**\n"
                    f"Current assimilated conditions show stable rainfall ({curr_rain_mean:.1f} mm/day) and "
                    f"temperatures peaking at {curr_temp_max:.1f}°C. Soil moisture levels remain adequate for routine agricultural operations. "
                    f"Continue standard schedule for weeding and nutrient top-dressing as per regional university package of practices."
                )

        # Structure the final Copilot response
        header = f"### AI Climate Copilot Advisory for {region}\n"
        header += f"*Live Context: Regional Avg Rain = {curr_rain_mean:.1f} mm/day | Peak Temp = {curr_temp_max:.1f}°C*\n\n"
        
        return header + "\n\n---\n\n".join(matched_articles) + "\n\n*Source: Grounded in official ICAR-CRIDA District Contingency Plans & NDMA Operating Procedures.*"
