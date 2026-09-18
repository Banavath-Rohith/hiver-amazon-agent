"""Pytest test suite for the AmazonHelp AI customer-support agent.

Run from the project root:
    python -m pytest tests/ -v

Tests cover:
  - intent_classifier: correct intent assignment, confidence range, non-English fallback
  - retrieval: returns list, respects limit, handles empty query
  - escalation: sensitive keyword rule, low confidence rule, no evidence rule
  - agent.run(): correct schema, AUTO case, ESCALATE case
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from intent_classifier import classify, tokens
from retrieval import retrieve
from escalation import decide
from agent import run
from reply_generator import draft_reply


# ─── intent_classifier ────────────────────────────────────────────────────────

class TestIntentClassifier:
    def test_delivery_intent(self):
        result = classify("@AmazonHelp my package has not arrived and tracking shows delayed")
        assert result["intent"] == "delivery_or_order"

    def test_account_intent(self):
        result = classify("@AmazonHelp I can't log in, my password reset email never came")
        assert result["intent"] == "account_or_login"

    def test_payment_intent(self):
        result = classify("@AmazonHelp I was charged twice for the same order, need a refund")
        assert result["intent"] == "payment_or_refund"

    def test_product_intent(self):
        result = classify("@AmazonHelp the Kindle device I received is broken and damaged")
        assert result["intent"] == "product_or_service"

    def test_technical_intent(self):
        result = classify("@AmazonHelp the app keeps crashing and the website gives an error")
        assert result["intent"] == "technical_issue"

    def test_non_english_fallback(self):
        result = classify("@AmazonHelp 私の注文が届きません、助けてください")
        assert result["intent"] == "other_or_unclear"

    def test_empty_fallback(self):
        result = classify("")
        assert result["intent"] == "other_or_unclear"

    def test_confidence_range(self):
        result = classify("@AmazonHelp my order tracking shows delayed delivery status")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_triggered_keywords(self):
        result = classify("@AmazonHelp my order and tracking are not updated")
        assert isinstance(result["triggered"], list)
        assert all(isinstance(t, str) for t in result["triggered"])

    def test_tokens_returns_set(self):
        t = tokens("Hello World hello")
        assert isinstance(t, set)
        assert "hello" in t
        assert "world" in t


# ─── retrieval ────────────────────────────────────────────────────────────────

class TestRetrieval:
    def test_returns_list(self):
        results = retrieve("order not delivered tracking delayed")
        assert isinstance(results, list)

    def test_respects_limit(self):
        results = retrieve("order not delivered tracking delayed", limit=2)
        assert len(results) <= 2

    def test_result_schema(self):
        results = retrieve("package has not arrived")
        if results:
            for r in results:
                assert "message" in r
                assert "support_response" in r
                assert "similarity" in r
                assert 0.0 <= r["similarity"] <= 1.0

    def test_empty_query_returns_list(self):
        results = retrieve("")
        assert isinstance(results, list)

    def test_similarity_sorted_descending(self):
        results = retrieve("order tracking delayed delivered package")
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i+1]["similarity"]


# ─── escalation ───────────────────────────────────────────────────────────────

class TestEscalation:
    def _clf(self, intent, confidence, message=""):
        return {"intent": intent, "confidence": confidence, "message": message}

    def test_sensitive_keyword_escalates(self):
        result = decide(self._clf("payment_or_refund", 0.8, "I need a refund for my card charge"), [])
        assert result["action"] == "ESCALATE"

    def test_account_keyword_escalates(self):
        result = decide(self._clf("account_or_login", 0.8, "my account password was reset"), [])
        assert result["action"] == "ESCALATE"

    def test_unclear_intent_escalates(self):
        result = decide(self._clf("other_or_unclear", 0.0), [{"support_response": "test"}])
        assert result["action"] == "ESCALATE"

    def test_technical_escalates(self):
        result = decide(self._clf("technical_issue", 0.8, "the app has an error"), [{"support_response": "test"}])
        assert result["action"] == "ESCALATE"

    def test_low_confidence_escalates(self):
        result = decide(self._clf("delivery_or_order", 0.05, "where thing"), [{"support_response": "test"}])
        assert result["action"] == "ESCALATE"

    def test_no_evidence_escalates(self):
        result = decide(self._clf("delivery_or_order", 0.8, "my order is late"), [])
        assert result["action"] == "ESCALATE"

    def test_auto_when_all_checks_pass(self):
        clf = self._clf("delivery_or_order", 0.9, "order has not arrived")
        ev  = [{"support_response": "We will look into this for you."}]
        result = decide(clf, ev)
        assert result["action"] == "AUTO"

    def test_decision_has_reason(self):
        clf = self._clf("delivery_or_order", 0.9, "order delayed")
        ev  = [{"support_response": "Our team will investigate."}]
        result = decide(clf, ev)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 10


# ─── agent ────────────────────────────────────────────────────────────────────

class TestAgent:
    REQUIRED_KEYS = {"intent", "confidence", "triggered", "action", "reason", "reply", "evidence"}

    def test_output_schema(self):
        result = run("@AmazonHelp my order has not arrived, tracking shows delayed")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_intent_is_string(self):
        result = run("Where is my delivery?")
        assert isinstance(result["intent"], str)

    def test_confidence_is_float(self):
        result = run("I need a refund for my order")
        assert isinstance(result["confidence"], float)

    def test_action_is_valid(self):
        result = run("My package was supposed to arrive today but tracking shows delayed")
        assert result["action"] in {"AUTO", "ESCALATE"}

    def test_reply_is_string(self):
        result = run("When will my order arrive?")
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 0

    def test_evidence_is_list(self):
        result = run("My order says delivered but I never got it")
        assert isinstance(result["evidence"], list)

    def test_sensitive_message_escalates(self):
        result = run("@AmazonHelp I need a refund my card was charged twice")
        assert result["action"] == "ESCALATE"


# ─── reply_generator ──────────────────────────────────────────────────────────

class TestReplyGenerator:
    def test_no_evidence_returns_fallback(self):
        reply = draft_reply("my order is late", [])
        assert "specialist" in reply.lower() or "support" in reply.lower()

    def test_with_evidence_returns_response(self):
        ev = [{"support_response": "We will look into this right away for you."}]
        reply = draft_reply("order delayed", ev)
        assert reply == "We will look into this right away for you."

    def test_empty_response_in_evidence_falls_back(self):
        ev = [{"support_response": ""}]
        reply = draft_reply("order delayed", ev)
        assert isinstance(reply, str)
        assert len(reply) > 0
