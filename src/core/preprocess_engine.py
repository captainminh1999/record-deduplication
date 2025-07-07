"""
Core preprocessing engine - pure business logic with no I/O or terminal output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd

from ..corp_designators import CORP_PREFIXES, CORP_SUFFIXES
from .openai_engine import OpenAIEngine
from .openai_types import OpenAIConfig


# Compiled regex patterns for efficiency
_PREFIX_RE = re.compile(rf"^(?:{'|'.join(CORP_PREFIXES)})\b[\s\.,]*", flags=re.I)
_SUFFIX_RE = re.compile(rf"\b(?:{'|'.join(CORP_SUFFIXES)})\b\.?", flags=re.I)
_PUNCT_RE = re.compile(r"[^\w\s&/\.-]+")
_WS_RE = re.compile(r"\s+")


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""
    use_openai: bool = False
    openai_model: str = "gpt-4o-mini"
    remove_duplicates: bool = True


@dataclass
class PreprocessStats:
    """Statistics from preprocessing operation."""
    initial_rows: int
    final_rows: int
    unique_companies_before: int
    unique_companies_after: int
    missing_company: int
    missing_domain: int
    domain_key: Optional[str]
    duplicates_removed: int
    dup_by_company: int
    dup_by_domain: int
    empty_company_names: int


@dataclass
class PreprocessResult:
    """Result of preprocessing operation."""
    cleaned_df: pd.DataFrame
    duplicates_df: pd.DataFrame
    stats: PreprocessStats


def normalize_company_name(name: str) -> str:
    """Return a cleaned company name following a set of rules."""
    if not name:
        return ""
    text_norm = unicodedata.normalize("NFKD", str(name))
    ascii_text = text_norm.encode("ascii", "ignore").decode("ascii")
    if not ascii_text.strip():
        if "ソニー" in name:
            ascii_text = "sony"
    text = ascii_text.lower()
    text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"^the\s+", "", text)
    text = _PREFIX_RE.sub("", text).strip()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    for _ in range(2):
        text = _SUFFIX_RE.sub("", text).strip()
        text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"[\./-]+$", "", text).strip()
    return text


def _normalize_domain(domain: str) -> str:
    """Return a cleaned domain name."""
    if not domain or pd.isna(domain):
        return ""
    domain_str = str(domain).lower().strip()
    if "@" in domain_str:
        domain_str = domain_str.split("@")[-1]
    domain_str = re.sub(r"^https?://(?:www\.)?", "", domain_str)
    domain_str = re.sub(r"^www\.", "", domain_str)
    domain_str = domain_str.split("/")[0]
    if not re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", domain_str):
        return ""
    return domain_str


def _normalize_phone(phone: str) -> str:
    """Return a cleaned phone number."""
    if not phone or pd.isna(phone):
        return ""
    phone_str = re.sub(r"[^\d]", "", str(phone))
    if len(phone_str) == 10:
        return f"({phone_str[:3]}) {phone_str[3:6]}-{phone_str[6:]}"
    elif len(phone_str) == 11 and phone_str[0] == "1":
        return f"1-({phone_str[1:4]}) {phone_str[4:7]}-{phone_str[7:]}"
    return ""


def _normalize_address(address_parts: List[str]) -> str:
    """Combine and normalize address parts."""
    combined = " ".join(str(part) for part in address_parts if pd.notna(part) and str(part).strip())
    if not combined.strip():
        return ""
    normalized = re.sub(r"\s+", " ", combined.strip())
    return normalized


class PreprocessEngine:
    """Core preprocessing business logic."""
    
    def process(self, df: pd.DataFrame, config: PreprocessConfig) -> PreprocessResult:
        """
        Process the dataframe with the given configuration.
        
        This is pure business logic - no file I/O, no terminal output.
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        initial_rows = len(df)
        
        # Auto-detect columns and create company_clean
        company_key = self._detect_company_column(df)
        if not company_key:
            raise ValueError("No company column found")
            
        domain_key = self._detect_domain_column(df)
        phone_key = self._detect_phone_column(df)
        address_keys = self._detect_address_columns(df)
        
        # Set record_id as index if it exists, otherwise create it
        if "record_id" in df.columns:
            df = df.set_index("record_id")
        else:
            df.index.name = "record_id"
        
        # Track original unique companies
        unique_companies_before = df[company_key].nunique()
        
        # Clean company names
        df["company_clean"] = df[company_key].apply(normalize_company_name)
        
        # Optional OpenAI translation
        if config.use_openai:
            non_empty_companies = df[df["company_clean"].notna() & (df["company_clean"] != "")]["company_clean"].unique()
            if len(non_empty_companies) > 0:
                # Use modular OpenAI engine for translation
                engine = OpenAIEngine()
                openai_config = OpenAIConfig(model=config.openai_model)
                result = engine.translate_to_english(non_empty_companies, openai_config)
                translated = result.translations
                translation_map = dict(zip(non_empty_companies, translated))
                df["company_clean"] = df["company_clean"].map(translation_map).fillna(df["company_clean"])
        
        # Clean other fields
        if domain_key:
            df["domain_clean"] = df[domain_key].apply(_normalize_domain)
        
        if phone_key:
            df["phone_clean"] = df[phone_key].apply(_normalize_phone)
        
        if address_keys:
            df["address_clean"] = df[address_keys].apply(
                lambda row: _normalize_address(row.tolist()), axis=1
            )
        
        # Calculate stats
        unique_companies_after = df["company_clean"].nunique()
        missing_company = df["company_clean"].isna().sum() + (df["company_clean"] == "").sum()
        missing_domain = df["domain_clean"].isna().sum() + (df["domain_clean"] == "").sum() if domain_key else initial_rows
        empty_company_names = (df["company_clean"] == "").sum()
        
        # Handle duplicates
        duplicates_df = pd.DataFrame()
        dup_by_company = 0
        dup_by_domain = 0
        
        if config.remove_duplicates:
            # Create unique identifier for deduplication
            df["_id_str"] = df["company_clean"].astype(str)
            if domain_key and "domain_clean" in df.columns:
                df["_id_str"] += "|" + df["domain_clean"].astype(str)
            
            # Find duplicates
            dup_mask = df.duplicated(subset=["_id_str"], keep="first")
            duplicates_df = df[dup_mask].copy()
            
            # Calculate duplicate stats
            company_dup_mask = df.duplicated(subset=["company_clean"], keep="first")
            dup_by_company = company_dup_mask.sum()
            
            if domain_key and "domain_clean" in df.columns:
                domain_dup_mask = df.duplicated(subset=["domain_clean"], keep="first")
                dup_by_domain = domain_dup_mask.sum()
            
            # Remove duplicates
            df = df[~dup_mask].drop(columns=["_id_str"])
        
        # Create stats
        stats = PreprocessStats(
            initial_rows=initial_rows,
            final_rows=len(df),
            unique_companies_before=unique_companies_before,
            unique_companies_after=unique_companies_after,
            missing_company=missing_company,
            missing_domain=missing_domain,
            domain_key=domain_key,
            duplicates_removed=len(duplicates_df),
            dup_by_company=dup_by_company,
            dup_by_domain=dup_by_domain,
            empty_company_names=empty_company_names
        )
        
        return PreprocessResult(
            cleaned_df=df,
            duplicates_df=duplicates_df,
            stats=stats
        )
    
    def _detect_company_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the company column."""
        possible_names = ["company", "company_name", "organization", "org", "business", "firm"]
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        # If no exact match, look for columns containing these terms
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        return None
    
    def _detect_domain_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the domain/website column."""
        possible_names = ["domain", "website", "url", "web", "site", "email"]
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        return None
    
    def _detect_phone_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the phone column."""
        possible_names = ["phone", "telephone", "tel", "mobile", "cell"]
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                return col
        return None
    
    def _detect_address_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect address-related columns."""
        address_cols = []
        possible_names = ["address", "street", "city", "state", "zip", "postal", "location"]
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                address_cols.append(col)
        return address_cols
