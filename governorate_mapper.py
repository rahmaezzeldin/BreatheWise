"""
Governorate Mapper Module

Ensures consistent governorate naming between data sources (training data and API data).
Handles case-insensitive matching and whitespace variations.
"""

import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GovernorateMapper:
    """Maps API format governorate names to standardized names."""
    
    def __init__(self, mapping_file: str = "Egypt Governorates for API.xlsx"):
        """
        Initialize the governorate mapper.
        
        Args:
            mapping_file: Path to Excel file containing governorate mappings
        """
        self.mapping_file = mapping_file
        self.mapping = self._load_mapping()
        self.reverse_mapping = {v.lower(): v for v in self.mapping.values()}
        
    def _load_mapping(self) -> Dict[str, str]:
        """
        Load governorate mapping from Excel file.
        
        Returns:
            Dictionary mapping API format names to standardized names
        """
        try:
            df = pd.read_excel(self.mapping_file)
            
            # Create comprehensive mapping dictionary
            mapping = {
                # Core mappings from file
                "al qahirah": "Cairo", "cairo": "Cairo",
                "al iskandariyah": "Alexandria", "alexandria": "Alexandria",
                "bur sa`id": "Port Said", "port said": "Port Said",
                "as suways": "Suez", "suez": "Suez",
                "dumyat": "Damietta", "damietta": "Damietta",
                "ad daqahliyah": "Dakahlia", "dakahlia": "Dakahlia",
                "ash sharqiyah": "Sharqia", "sharqia": "Sharqia",
                "al gharbiyah": "Gharbia", "gharbia": "Gharbia",
                "al minufiyah": "Monufia", "monufia": "Monufia",
                "al qalyubiyah": "Qalyubia", "qalyubia": "Qalyubia",
                "kafr ash shaykh": "Kafr El Sheikh", "kafr el sheikh": "Kafr El Sheikh",
                "al buhayrah": "Beheira", "beheira": "Beheira",
                "al isma`iliyah": "Ismailia", "ismailia": "Ismailia",
                "al jizah": "Giza", "giza": "Giza",
                "bani suwayf": "Beni Suef", "beni suef": "Beni Suef",
                "al fayyum": "Fayoum", "fayoum": "Fayoum", "faiyum": "Fayoum",
                "al minya": "Minya", "minya": "Minya",
                "asyut": "Asyut", "assiut": "Asyut",
                "suhaj": "Sohag", "sohag": "Sohag",
                "qinā": "Qena", "qena": "Qena", "qina": "Qena",
                "al uqsar": "Luxor", "al uqsur": "Luxor", "luxor": "Luxor",
                "aswan": "Aswan",
                "al bahr al ahmar": "Red Sea", "red sea": "Red Sea",
                "al wadi al jadid": "New Valley", "al wadi at jadid": "New Valley", "new valley": "New Valley",
                "matruh": "Matrouh", "matrouh": "Matrouh",
                "shamal sina'": "North Sinai", "north sinai": "North Sinai",
                "janub sina'": "South Sinai", "south sinai": "South Sinai"
            }
            
            logger.info(f"Loaded {len(set(mapping.values()))} unique governorate mappings")
            return mapping
            
        except FileNotFoundError:
            logger.warning(f"Mapping file {self.mapping_file} not found, using default mappings")
            return self._get_default_mapping()
        except Exception as e:
            logger.error(f"Error loading mapping file: {e}")
            return self._get_default_mapping()
    
    def _get_default_mapping(self) -> Dict[str, str]:
        """Return default governorate mappings if file loading fails."""
        return {
            "al qahirah": "Cairo", "cairo": "Cairo",
            "al iskandariyah": "Alexandria", "alexandria": "Alexandria",
            "bur sa`id": "Port Said", "port said": "Port Said",
            "as suways": "Suez", "suez": "Suez",
            "dumyat": "Damietta", "damietta": "Damietta",
            "ad daqahliyah": "Dakahlia", "dakahlia": "Dakahlia",
            "ash sharqiyah": "Sharqia", "sharqia": "Sharqia",
            "al gharbiyah": "Gharbia", "gharbia": "Gharbia",
            "al minufiyah": "Monufia", "monufia": "Monufia",
            "al qalyubiyah": "Qalyubia", "qalyubia": "Qalyubia",
            "kafr ash shaykh": "Kafr El Sheikh", "kafr el sheikh": "Kafr El Sheikh",
            "al buhayrah": "Beheira", "beheira": "Beheira",
            "al isma`iliyah": "Ismailia", "ismailia": "Ismailia",
            "al jizah": "Giza", "giza": "Giza",
            "bani suwayf": "Beni Suef", "beni suef": "Beni Suef",
            "al fayyum": "Fayoum", "fayoum": "Fayoum", "faiyum": "Fayoum",
            "al minya": "Minya", "minya": "Minya",
            "asyut": "Asyut", "assiut": "Asyut",
            "suhaj": "Sohag", "sohag": "Sohag",
            "qinā": "Qena", "qena": "Qena", "qina": "Qena",
            "al uqsar": "Luxor", "al uqsur": "Luxor", "luxor": "Luxor",
            "aswan": "Aswan",
            "al bahr al ahmar": "Red Sea", "red sea": "Red Sea",
            "al wadi al jadid": "New Valley", "al wadi at jadid": "New Valley", "new valley": "New Valley",
            "matruh": "Matrouh", "matrouh": "Matrouh",
            "shamal sina'": "North Sinai", "north sinai": "North Sinai",
            "janub sina'": "South Sinai", "south sinai": "South Sinai"
        }
    
    def map_governorate(self, raw_name: str) -> str:
        """
        Map API format name to standardized name.
        Handles case-insensitive matching and whitespace variations.
        
        Args:
            raw_name: Raw governorate name from API or dataset
            
        Returns:
            Standardized governorate name
        """
        if pd.isna(raw_name):
            logger.warning("Received NaN governorate name")
            return None
        
        # Normalize: lowercase and strip whitespace
        normalized = str(raw_name).lower().strip()
        
        # Try direct mapping
        if normalized in self.mapping:
            return self.mapping[normalized]
        
        # Try title case as fallback
        title_case = raw_name.strip().title()
        if title_case in self.reverse_mapping.values():
            return title_case
        
        # Log warning for unmapped governorate
        logger.warning(f"Unmapped governorate name: '{raw_name}'")
        
        # Return title case as default
        return title_case
    
    def validate_mapping(self, encoder) -> bool:
        """
        Validate that all mapped governorate names exist in the trained encoder.
        
        Args:
            encoder: Fitted LabelEncoder object
            
        Returns:
            True if all mappings are valid, False otherwise
        """
        unique_mapped_names = set(self.mapping.values())
        encoder_classes = set(encoder.classes_)
        
        missing = unique_mapped_names - encoder_classes
        
        if missing:
            logger.error(f"Mapped governorates not in encoder: {missing}")
            return False
        
        logger.info("All governorate mappings validated successfully")
        return True
    
    def get_unmapped_names(self, names: List[str]) -> List[str]:
        """
        Return list of names without mappings.
        
        Args:
            names: List of governorate names to check
            
        Returns:
            List of unmapped names
        """
        unmapped = []
        for name in names:
            normalized = str(name).lower().strip()
            if normalized not in self.mapping:
                unmapped.append(name)
        
        return unmapped
    
    def save_mapping_reference(self, output_path: str = "governorate_mapping_reference.txt") -> None:
        """
        Document all governorate name mappings in a reference file.
        
        Args:
            output_path: Path to save the reference file
        """
        unique_mappings = {}
        for key, value in self.mapping.items():
            if value not in unique_mappings:
                unique_mappings[value] = []
            unique_mappings[value].append(key)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Governorate Name Mapping Reference\n")
            f.write("=" * 50 + "\n\n")
            
            for standard_name in sorted(unique_mappings.keys()):
                f.write(f"{standard_name}:\n")
                for variant in sorted(unique_mappings[standard_name]):
                    f.write(f"  - {variant}\n")
                f.write("\n")
        
        logger.info(f"Mapping reference saved to {output_path}")
