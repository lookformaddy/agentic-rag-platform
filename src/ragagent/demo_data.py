"""Synthetic demo corpus, glossary and golden set (all fictional).

The corpus describes "Northwind Outfitters", a fictional outdoor-retail
company invented for this project. No real company, product, or person
appears here; every fact is synthetic and safe to ship in a public repo.
"""
from __future__ import annotations

from ragagent.models import Document, EvalCase

GLOSSARY: dict[str, str] = {
    "warranty": "A warranty is a 1-year promise that covers manufacturing "
                "defects in materials and workmanship, excluding normal "
                "wear and tear.",
    "waterproof": "Waterproof means a product is built to keep water out "
                  "under rain and splashes, rated for outdoor use.",
    "loyalty points": "Loyalty points are rewards earned at 1 point per "
                      "dollar spent, redeemable for discounts once a "
                      "member reaches a tier threshold.",
    "express shipping": "Express shipping is a 2-day delivery service "
                        "that costs 12.99 dollars per order.",
    "return window": "The return window is the 30-day period after "
                     "delivery during which unused items can be returned "
                     "for a refund.",
}


def demo_documents() -> list[Document]:
    """Eight synthetic documents covering returns, shipping, loyalty,
    products, warranty, store hours and discounts."""
    return [
        Document(
            doc_id="returns-policy",
            title="Returns Policy",
            text=(
                "The return window is 30 days from the date of delivery. "
                "Items must be unused and carry their original tags to qualify "
                "for a return. Refunds are issued to the original payment "
                "method within 5 business days of the returned item arriving "
                "at our warehouse. Exchanges are always free and do not count "
                "against the 30-day return window. Sale items can be returned "
                "for store credit only."
            ),
            tags=("returns", "refund", "policy")),
        Document(
            doc_id="shipping-policy",
            title="Shipping Policy",
            text=(
                "Standard shipping costs $4.99 per order and arrives in 3 to 5 "
                "business days. Standard shipping is free on all orders over "
                "$75. Express shipping costs $12.99 per order and arrives in "
                "2 business days. Orders placed before 2 PM ship the same day. "
                "We ship to all 50 states and offer tracking on every order."
            ),
            tags=("shipping", "delivery", "policy")),
        Document(
            doc_id="loyalty-program",
            title="Northwind Rewards Loyalty Program",
            text=(
                "Members earn 1 point for every dollar spent at Northwind "
                "Outfitters. Silver tier starts at 500 points and gives 5% "
                "back on every purchase. Gold tier starts at 1500 points and "
                "gives 10% back on every purchase. Loyalty points expire after "
                "12 months of inactivity. Points can be redeemed at checkout "
                "for discounts on future orders."
            ),
            tags=("loyalty", "points", "rewards")),
        Document(
            doc_id="product-trail-jacket",
            title="Trailblazer Waterproof Jacket",
            text=(
                "The Trailblazer jacket costs $129.99 and is fully waterproof "
                "for rain and snow. It is available in sizes XS to XXL and in "
                "navy and forest colors. The jacket weighs 420 grams and packs "
                "into its own pocket for travel. It carries a 1-year warranty "
                "against manufacturing defects."
            ),
            tags=("product", "jacket", "price")),
        Document(
            doc_id="product-hiking-boots",
            title="Summit Hiking Boots",
            text=(
                "The Summit hiking boots cost $159.99 and are made from "
                "waterproof leather. They are available in sizes 7 to 13 and "
                "include a grippy rubber outsole for rocky trails. The boots "
                "weigh 610 grams per pair and carry a 1-year warranty against "
                "manufacturing defects."
            ),
            tags=("product", "boots", "price")),
        Document(
            doc_id="warranty-policy",
            title="Warranty Policy",
            text=(
                "Every Northwind product carries a 1-year warranty that covers "
                "manufacturing defects in materials and workmanship. The "
                "warranty does not cover normal wear and tear, accidental "
                "damage, or misuse. To claim under the warranty, contact "
                "support with your order number and photos of the defect. "
                "Approved claims are repaired or replaced free of charge."
            ),
            tags=("warranty", "policy")),
        Document(
            doc_id="store-hours",
            title="Store Hours",
            text=(
                "Northwind stores are open Monday to Saturday from 9 AM to "
                "9 PM. On Sunday stores open at 10 AM and close at 6 PM. "
                "Holiday hours may differ and are posted on the store door "
                "and on our website. Customer support by phone is available "
                "from 8 AM to 8 PM every day."
            ),
            tags=("store", "hours")),
        Document(
            doc_id="discounts-policy",
            title="Discounts Policy",
            text=(
                "Students receive a 10% discount with a valid student ID. "
                "The student discount cannot be combined with other offers "
                "or promotions. Sale items are excluded from all discounts. "
                "Loyalty members can combine points redemption with the "
                "student discount at checkout."
            ),
            tags=("discount", "student", "policy")),
    ]


def golden_set() -> list[EvalCase]:
    """Ten golden cases: factual, multi-part, tool-backed, and abstain."""
    return [
        EvalCase(case_id="returns-window",
                 question="What is the return window?",
                 expected_keywords=["30", "days"],
                 expected_sources=["returns-policy"]),
        EvalCase(case_id="shipping-cost",
                 question="How much does standard shipping cost?",
                 expected_keywords=["4.99"],
                 expected_sources=["shipping-policy"]),
        EvalCase(case_id="loyalty-points",
                 question="How many points do I earn per dollar spent?",
                 expected_keywords=["1 point", "dollar"],
                 expected_sources=["loyalty-program"]),
        EvalCase(case_id="multi-part",
                 question="What is the return window and how much does "
                          "express shipping cost?",
                 expected_keywords=["30", "12.99"],
                 expected_sources=["returns-policy", "shipping-policy"]),
        EvalCase(case_id="arithmetic-total",
                 question="If I buy the trail jacket at $129.99 and the "
                          "hiking boots at $159.99, what is the total "
                          "before tax?",
                 expected_keywords=["289.98"],
                 expected_sources=["product-trail-jacket",
                                   "product-hiking-boots"]),
        EvalCase(case_id="date-weekday",
                 question="What day of the week is 2026-11-17?",
                 expected_keywords=["Tuesday"],
                 expected_sources=[]),
        EvalCase(case_id="definition-warranty",
                 question="What is a warranty?",
                 expected_keywords=["1-year", "defects"],
                 expected_sources=["warranty-policy"]),
        EvalCase(case_id="abstain-revenue",
                 question="What is the quarterly revenue of Northwind "
                          "Outfitters?",
                 expected_keywords=[],
                 expected_sources=[],
                 expect_abstain=True),
        EvalCase(case_id="store-sunday",
                 question="What time do stores open on Sunday?",
                 expected_keywords=["10 AM"],
                 expected_sources=["store-hours"]),
        EvalCase(case_id="discount-combine",
                 question="Can I combine the student discount with other offers?",
                 expected_keywords=["cannot", "combined"],
                 expected_sources=["discounts-policy"]),
    ]
