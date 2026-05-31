# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
import time
import base64
from typing import List, Optional, Dict, Any
from datetime import datetime

import requests

from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit
from intergrax.websearch.providers.base import WebSearchProvider


class RedditAPIProvider(WebSearchProvider):
    """
    Full-featured Reddit search provider using official OAuth2 API.

    Auth:
      - Uses application-only OAuth2 (client_credentials).
      - Env vars:
          REDDIT_CLIENT_ID
          REDDIT_CLIENT_SECRET
          REDDIT_USER_AGENT

    Endpoints:
      - Auth:   https://www.reddit.com/api/v1/access_token
      - Search: https://oauth.reddit.com/search
      - Post + comments: https://oauth.reddit.com/comments/{id}.json

    Features:
      - Full search (title + body).
      - Rich post metadata (score, num_comments, upvote_ratio, nsfw, etc.).
      - Optional: fetch top-level comments for each post.

    Limitations:
      - Language filter: Reddit nie ma natywnego filtra językowego → ignorujemy spec.language.
      - Freshness: mapujemy spec.freshness (np. "day", "week", "month") na parametr `t`.
    """

    name: str = "reddit_api"

    _AUTH_URL: str = "https://www.reddit.com/api/v1/access_token"
    _API_BASE: str = "https://oauth.reddit.com"
    _SEARCH_ENDPOINT: str = f"{_API_BASE}/search"
    _COMMENTS_ENDPOINT_TEMPLATE: str = f"{_API_BASE}/comments/{{id}}.json"

    _DEFAULT_TIMEOUT: int = 20
    _MAX_LIMIT: int = 50  # Reddit pozwala do 100, ale ograniczamy dla stabilności.

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: Optional[int] = None,
        include_comments: bool = True,
        comments_limit: int = 10,
        default_freshness: Optional[str] = None,  # "hour", "day", "week", "month", "year", "all"
    ) -> None:
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "IntegraxWebSearch/1.0 (by intergrax.ai)")

        if not self.client_id:
            raise ValueError("RedditAPIProvider: missing client_id (REDDIT_CLIENT_ID).")
        if not self.client_secret:
            raise ValueError("RedditAPIProvider: missing client_secret (REDDIT_CLIENT_SECRET).")

        self.session = session or requests.Session()
        self.timeout = int(timeout or self._DEFAULT_TIMEOUT)
        self.include_comments = include_comments
        self.comments_limit = comments_limit
        self.default_freshness = default_freshness

        # Auth state
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # Required UA header for Reddit
        self.session.headers.update({"User-Agent": self.user_agent})

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_language": False,
            "supports_freshness": True,   # via 't' param in search
            "max_page_size": self._MAX_LIMIT,
        }

    # ------------------------------------------------------------------
    # OAuth2 handling
    # ------------------------------------------------------------------
    def _ensure_token(self) -> None:
        """
        Ensure we have a valid access token (client_credentials).
        Refresh if expired or missing.
        """
        now = time.time()
        if self._access_token and now < self._token_expires_at - 30:  # 30s safety margin
            return

        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": self.user_agent}

        try:
            resp = self.session.post(
                self._AUTH_URL,
                auth=auth,
                data=data,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            # Fail fast 
            self._access_token = None
            self._token_expires_at = 0
            raise

        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)

        if not token:
            raise RuntimeError("RedditAPIProvider: missing access_token in auth response.")

        self._access_token = token
        self._token_expires_at = now + int(expires_in)

        self.session.headers.update({
            "Authorization": f"bearer {self._access_token}",
            "User-Agent": self.user_agent,
        })

    # ------------------------------------------------------------------
    # Param building
    # ------------------------------------------------------------------
    def _build_search_params(self, spec: QuerySpec) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "q": spec.normalized_query(),
            "limit": spec.capped_top_k(self._MAX_LIMIT),
            "sort": "relevance",
            "restrict_sr": "false",
        }

        # Freshness: spec.freshness → t
        # Reddit options: hour, day, week, month, year, all
        freshness = getattr(spec, "freshness", None) or self.default_freshness
        if freshness:
            params["t"] = freshness

        return params

    # ------------------------------------------------------------------
    # Comments fetch
    # ------------------------------------------------------------------
    def _fetch_comments(self, post_id: str) -> List[Dict[str, Any]]:
        """
        Fetch top-level comments for a given post.

        Returns a simplified list of comment dicts:
          {
            "author": str,
            "body": str,
            "score": int,
            "created_utc": float,
            "num_replies": int,
          }
        """
        if not self.include_comments or not post_id:
            return []

        url = self._COMMENTS_ENDPOINT_TEMPLATE.format(id=post_id)
        params = {
            "limit": self.comments_limit,
            "depth": 1,
        }

        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []

        # comments data is typically in data[1]["data"]["children"]
        comments_block = []
        if isinstance(data, list) and len(data) > 1:
            comments_block = data[1].get("data", {}).get("children", [])

        comments: List[Dict[str, Any]] = []
        for c in comments_block:
            cdata = c.get("data") if isinstance(c, dict) else None
            if not cdata:
                continue
            if cdata.get("body") is None:
                continue

            created_utc = cdata.get("created_utc")
            num_replies = len(cdata.get("replies", {}).get("data", {}).get("children", [])) \
                if isinstance(cdata.get("replies"), dict) else 0

            comments.append({
                "id": cdata.get("id"),
                "author": cdata.get("author"),
                "body": cdata.get("body"),
                "score": cdata.get("score"),
                "created_utc": created_utc,
                "num_replies": num_replies,
            })

        return comments

    # ------------------------------------------------------------------
    # Mapping to SearchHit
    # ------------------------------------------------------------------
    def _to_hit(self, child: Dict[str, Any], spec: QuerySpec, rank: int) -> Optional[SearchHit]:
        data = child.get("data")
        if not data:
            return None

        title = data.get("title") or ""
        permalink = data.get("permalink") or ""
        url = f"https://www.reddit.com{permalink}" if permalink else None

        if not url:
            return None

        snippet = data.get("selftext") or None
        subreddit = data.get("subreddit") or None
        author = data.get("author") or None

        created_utc = data.get("created_utc")
        published_at: Optional[datetime] = None
        if created_utc:
            try:
                published_at = datetime.utcfromtimestamp(float(created_utc))
            except Exception:
                published_at = None

        post_id = data.get("id")

        comments = self._fetch_comments(post_id) if self.include_comments else []

        extra: Dict[str, Any] = {
            "subreddit": subreddit,
            "author": author,
            "score": data.get("score"),
            "ups": data.get("ups"),
            "downs": data.get("downs"),
            "num_comments": data.get("num_comments"),
            "upvote_ratio": data.get("upvote_ratio"),
            "over_18": data.get("over_18"),
            "is_self": data.get("is_self"),
            "is_original_content": data.get("is_original_content"),
            "is_crosspostable": data.get("is_crosspostable"),
            "permalink": permalink,
            "reddit_id": post_id,
            "domain": data.get("domain"),
            "url_overridden_by_dest": data.get("url_overridden_by_dest"),
            "thumbnail": data.get("thumbnail"),
            "created_utc": created_utc,
            "comments": comments,
        }

        return SearchHit(
            provider=self.name,
            query_issued=spec.query,
            rank=rank,
            title=title,
            url=url,
            snippet=snippet,
            displayed_link=subreddit,
            published_at=published_at,
            source_type="reddit_post",
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------
    def search(self, spec: QuerySpec) -> List[SearchHit]:
        self._ensure_token()
        params = self._build_search_params(spec)

        try:
            r = self.session.get(self._SEARCH_ENDPOINT, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []

        children = data.get("data", {}).get("children", []) or []
        hits: List[SearchHit] = []

        for i, child in enumerate(children, start=1):
            hit = self._to_hit(child, spec, i)
            if hit:
                hits.append(hit)

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
