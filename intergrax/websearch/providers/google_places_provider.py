# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import List, Optional, Dict, Any
from datetime import datetime

import requests

from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit
from intergrax.websearch.providers.base import WebSearchProvider


class GooglePlacesProvider(WebSearchProvider):
    """
    Google Places / Google Business provider (Text Search + Details).

    Environment variables:
      GOOGLE_PLACES_API_KEY : API key for Google Places API

    Endpoints (Places API):
      - Text Search: https://maps.googleapis.com/maps/api/place/textsearch/json
      - Details:     https://maps.googleapis.com/maps/api/place/details/json

    Features:
      - Text search by arbitrary query (name + city, category, etc.).
      - Returns core business data:
          * name, address, location (lat/lng)
          * rating, user_ratings_total
          * types (categories)
          * website, international_phone_number, opening_hours
          * Google Maps URL (via url or constructed maps link)

    Notes:
      - Uses QuerySpec.query as textsearch query.
      - spec.language → Places "language"
      - spec.region   → Places "region"
      - Freshness is not applicable (ignored).
    """

    name: str = "google_places"

    _TEXT_SEARCH_ENDPOINT: str = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    _DETAILS_ENDPOINT: str = "https://maps.googleapis.com/maps/api/place/details/json"

    _DEFAULT_TIMEOUT: int = 20
    _MAX_PAGE_SIZE: int = 20           # textsearch default max=20
    _MAX_DETAILS_LOOKUPS: int = 10     # to limit extra calls / quotas

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: Optional[int] = None,
        fetch_details: bool = True,
        max_details_lookups: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_PLACES_API_KEY", "")
        if not self.api_key:
            raise ValueError("GooglePlacesProvider: missing API key (GOOGLE_PLACES_API_KEY).")

        self.session = session or requests.Session()
        self.timeout = int(timeout or self._DEFAULT_TIMEOUT)
        self.fetch_details = fetch_details
        self.max_details_lookups = int(max_details_lookups or self._MAX_DETAILS_LOOKUPS)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_language": True,    # via Places "language" param
            "supports_freshness": False,  # not applicable
            "max_page_size": self._MAX_PAGE_SIZE,
        }

    # ------------------------------------------------------------------
    # Parameter builders
    # ------------------------------------------------------------------
    def _build_textsearch_params(self, spec: QuerySpec) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "key": self.api_key,
            "query": spec.normalized_query(),     # np. "restauracja włoska Warszawa"
        }

        # language param – ISO code or BCP-47 (np. "pl", "pl-PL")
        if spec.language:
            params["language"] = spec.language
        elif spec.locale:
            # gdy user podaje locale "pl-PL", Places sobie poradzi
            params["language"] = spec.locale

        # region param (ccTLD) – np. "pl"
        if spec.region:
            # jeśli w QuerySpec masz "pl-PL", możesz chcieć wyciąć country
            region = spec.region
            if "-" in region:
                region = region.split("-")[-1]
            params["region"] = region.lower()

        # Places Text Search nie ma natywnego "top_k"; limit 20 na stronę.
        # Możesz ewentualnie trzymać target w spec.top_k i przycinać wyniki po stronie klienta.
        return params

    def _build_details_params(self, place_id: str, language: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "key": self.api_key,
            "place_id": place_id,
            # fields: ograniczamy do kluczowych pól, reszta to "extra"
            "fields": ",".join([
                "name",
                "formatted_address",
                "geometry/location",
                "international_phone_number",
                "website",
                "url",
                "opening_hours",
                "rating",
                "user_ratings_total",
                "types",
                "business_status",
            ]),
        }
        if language:
            params["language"] = language
        return params

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------
    def _fetch_place_details(
        self,
        place_id: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch extended details for a single place_id.
        Returns result dict (or empty dict on failure).
        """
        params = self._build_details_params(place_id, language=language)
        try:
            r = self.session.get(self._DETAILS_ENDPOINT, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return {}

        if data.get("status") != "OK":
            return {}

        return data.get("result") or {}

    def _to_hit(
        self,
        base_place: Dict[str, Any],
        spec: QuerySpec,
        rank: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[SearchHit]:
        """
        Map a Places TextSearch result (+ optional Details) to SearchHit.
        """
        details = details or {}

        # Prefer details.name, fallback to base_place.name
        name = details.get("name") or base_place.get("name") or ""
        if not name:
            return None

        # Address
        formatted_address = details.get("formatted_address") or base_place.get("formatted_address") or None

        # Website vs Maps URL
        website = details.get("website")
        maps_url = details.get("url")  # direct Google Maps link
        url = website or maps_url
        if not url:
            # Ostateczny fallback – możemy skonstruować link:
            #  https://www.google.com/maps/search/?api=1&query=lat,lng&query_place_id=PLACE_ID
            place_id = base_place.get("place_id") or details.get("place_id")
            loc = details.get("geometry", {}).get("location") or base_place.get("geometry", {}).get("location", {})
            lat = loc.get("lat")
            lng = loc.get("lng")
            if place_id and lat is not None and lng is not None:
                url = (
                    "https://www.google.com/maps/search/"
                    f"?api=1&query={lat},{lng}&query_place_id={place_id}"
                )

        # rating & reviews
        rating = details.get("rating", base_place.get("rating"))
        user_ratings_total = details.get("user_ratings_total", base_place.get("user_ratings_total"))

        # snippet: prosty opis łączący adres + rating
        snippet_parts: List[str] = []
        if formatted_address:
            snippet_parts.append(formatted_address)
        if rating is not None and user_ratings_total is not None:
            snippet_parts.append(f"Ocena: {rating} ({user_ratings_total} opinii)")
        snippet = " | ".join(snippet_parts) if snippet_parts else None

        # location
        geom = details.get("geometry", {}).get("location") or base_place.get("geometry", {}).get("location", {})
        lat = geom.get("lat")
        lng = geom.get("lng")

        # phone
        phone = details.get("international_phone_number")

        # opening hours
        opening_hours = details.get("opening_hours") or {}
        opening_weekday_text = opening_hours.get("weekday_text")

        # types / categories
        types = details.get("types") or base_place.get("types") or []

        extra: Dict[str, Any] = {
            "place_id": base_place.get("place_id") or details.get("place_id"),
            "formatted_address": formatted_address,
            "location": {"lat": lat, "lng": lng},
            "rating": rating,
            "user_ratings_total": user_ratings_total,
            "types": types,
            "business_status": details.get("business_status") or base_place.get("business_status"),
            "website": website,
            "google_maps_url": maps_url,
            "international_phone_number": phone,
            "opening_hours": opening_hours,
            "opening_weekday_text": opening_weekday_text,
        }

        # published_at – Places nie podaje daty "utworzenia" biznesu.
        # Można by kiedyś dodać heurystykę, ale na razie None.
        published_at: Optional[datetime] = None

        return SearchHit(
            provider=self.name,
            query_issued=spec.query,
            rank=rank,
            title=name,
            url=url or "",
            snippet=snippet,
            displayed_link=formatted_address,
            published_at=published_at,
            source_type="business_place",
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------
    def search(self, spec: QuerySpec) -> List[SearchHit]:
        params = self._build_textsearch_params(spec)

        try:
            r = self.session.get(self._TEXT_SEARCH_ENDPOINT, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return []

        results = data.get("results") or []
        if not results:
            return []

        # Opcjonalnie pobieramy szczegóły dla pierwszych N wyników, żeby wzbogacić dane.
        details_by_id: Dict[str, Dict[str, Any]] = {}
        if self.fetch_details:
            language = params.get("language")
            for place in results[: self.max_details_lookups]:
                place_id = place.get("place_id")
                if not place_id:
                    continue
                details = self._fetch_place_details(place_id, language=language)
                if details:
                    details_by_id[place_id] = details

        hits: List[SearchHit] = []
        for i, place in enumerate(results, start=1):
            place_id = place.get("place_id")
            details = details_by_id.get(place_id) if place_id else None
            hit = self._to_hit(place, spec, i, details=details)
            if hit:
                hits.append(hit)

        # Uporządkuj rank po ewentualnym filtrowaniu
        normalized_hits: List[SearchHit] = []
        for idx, h in enumerate(hits, start=1):
            if h.rank == idx:
                normalized_hits.append(h)
            else:
                normalized_hits.append(
                    SearchHit(
                        provider=h.provider,
                        query_issued=h.query_issued,
                        rank=idx,
                        title=h.title,
                        url=h.url,
                        snippet=h.snippet,
                        displayed_link=h.displayed_link,
                        published_at=h.published_at,
                        source_type=h.source_type,
                        extra=h.extra,
                    )
                )

        return normalized_hits

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
