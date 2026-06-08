"""
==============================================
XML UTILITIES FÜR MUJOCO KONFIGURATIONEN
==============================================
Enthält die XML-Verarbeitungsfunktionen aus dem Original-Skript.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os


def pretty_xml(element):
    """
    Formatiert XML-Elemente für bessere Lesbarkeit.
    Wird für das Speichern von Simulationskonfigurationen verwendet.
    
    Args:
        element: ET.Element - Das zu formatierende XML-Element
        
    Returns:
        str: Formatierter XML-String mit Einrückungen
    """
    raw_xml = ET.tostring(element, encoding="utf-8")
    parsed_xml = minidom.parseString(raw_xml)
    return parsed_xml.toprettyxml(indent="  ")  # Zwei Leerzeichen als Einrückung


def save_body_configuration(env, filepath="data/simulation_config/body_012b.xml"):
    """
    Speichert die aktuelle Body-Konfiguration (Positionen, Rotationen, Kräfte).
    Wird für Analyse und Debugging verwendet.
    
    Args:
        env: Environment-Objekt mit model und data
        filepath: Pfad für die Ausgabedatei
    """
    root_data_body = ET.Element("SimulationConfigurationBody")

    for i in range(env.model.nbody):
        body_elem = ET.SubElement(root_data_body, "Body", id=str(i), 
                                  name=env.model.body(i).name)
        
        pos_elem = ET.SubElement(body_elem, "Position")
        pos_elem.text = ", ".join(map(str, env.data.xpos[i]))

        rot_elem = ET.SubElement(body_elem, "Rotation")
        rot_elem.text = ", ".join(map(str, env.data.xmat[i]))

        force_elem = ET.SubElement(body_elem, "ExternalForces")
        force_elem.text = ", ".join(map(str, env.data.xfrc_applied[i][:3]))

        moment_elem = ET.SubElement(body_elem, "ExternalMoments")
        moment_elem.text = ", ".join(map(str, env.data.cfrc_ext[i][3:]))

        vel_elem = ET.SubElement(body_elem, "LinearVelocity")
        vel_elem.text = ", ".join(map(str, env.data.cvel[i]))

    # Verzeichnis erstellen falls nicht vorhanden
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    xml_str = pretty_xml(root_data_body)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"Sim_config gespeichert: {filepath}")


def save_geometry_configuration(env, filepath="data/simulation_config/geom_007.xml"):
    """
    Speichert die aktuelle Geometrie-Konfiguration der Simulation in XML.
    Nützlich für Debugging und Analyse der Körperpositionen.
    
    Args:
        env: Environment-Objekt mit model und data
        filepath: Pfad für die Ausgabedatei
    """
    root_data_geom = ET.Element("SimulationConfigurationGeom")

    for i in range(env.model.ngeom):
        geom_elem = ET.SubElement(root_data_geom, "Geometry", id=str(i), 
                                  name=env.model.geom(i).name, 
                                  type=str(env.model.geom(i).type))
        
        pos_elem = ET.SubElement(geom_elem, "Position")
        pos_elem.text = ", ".join(map(str, env.data.geom_xpos[i]))

        rot_elem = ET.SubElement(geom_elem, "Rotation")
        rot_elem.text = ", ".join(map(str, env.data.geom_xmat[i]))

    # Verzeichnis erstellen falls nicht vorhanden
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    xml_str = pretty_xml(root_data_geom)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"Sim_config gespeichert: {filepath}")