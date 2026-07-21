#!/usr/bin/env python3
"""Repair narrowly scoped accessibility-tag artifacts in the generated guide PDF.

The LuaLaTeX tagging pipeline currently emits two structures that Adobe Acrobat's
accessibility checker treats as failures:

* numbered table-of-contents entries contain ``Lbl`` descendants under ``TOCI``;
* some tables contain empty ``Artifact``/``Private`` branches whose descendants
  include ``TD``/``TH`` cells outside a ``TR``.

This script changes TOC ``Lbl`` roles to ``Span`` and removes only table artifact
branches that have no marked-content references or meaningful text metadata.
Unexpected nonempty or malformed structures cause validation to fail instead of
being silently changed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    IndirectObject,
    NameObject,
    NumberObject,
)

STRUCT_ELEM_TYPE = "/StructElem"
CONTENT_REFERENCE_TYPES = {"/MCR", "/OBJR"}
EMPTY_TABLE_WRAPPER_ROLES = {"/Artifact", "/Private"}
ALLOWED_EMPTY_TABLE_SUBTREE_ROLES = {
    "/Artifact",
    "/Private",
    "/TD",
    "/TH",
    "/Div",
    "/P",
    "/Span",
    "/text",
    "/text-unit",
}
MEANINGFUL_TEXT_KEYS = ("/Alt", "/ActualText", "/E", "/T")


class RepairError(RuntimeError):
    """Raised when the PDF structure is missing or unsafe to repair."""


@dataclass
class RepairStats:
    toc_labels_changed: int = 0
    table_wrappers_removed: int = 0
    orphan_cells_removed: int = 0
    id_tree_entries_removed: int = 0


def _resolve(value):
    return value.get_object() if isinstance(value, IndirectObject) else value


def _is_structure_element(value) -> bool:
    resolved = _resolve(value)
    return isinstance(resolved, DictionaryObject) and (
        resolved.get("/Type") == STRUCT_ELEM_TYPE or "/S" in resolved
    )


def _role(value) -> str:
    resolved = _resolve(value)
    if not isinstance(resolved, DictionaryObject):
        return ""
    return str(resolved.get("/S", ""))


def _children(value) -> list:
    resolved = _resolve(value)
    if not isinstance(resolved, DictionaryObject) or "/K" not in resolved:
        return []
    kids = resolved["/K"]
    if isinstance(kids, ArrayObject):
        return list(kids)
    return [kids]


def _iter_structure_children(value) -> Iterator:
    for child in _children(value):
        if _is_structure_element(child):
            yield child


def _subtree_roles(value) -> set[str]:
    roles = {_role(value)}
    for child in _iter_structure_children(value):
        roles.update(_subtree_roles(child))
    return roles


def _subtree_ids(value) -> set[str]:
    resolved = _resolve(value)
    ids: set[str] = set()
    if isinstance(resolved, DictionaryObject):
        element_id = resolved.get("/ID")
        if element_id is not None and str(element_id):
            ids.add(str(element_id))
    for child in _iter_structure_children(value):
        ids.update(_subtree_ids(child))
    return ids


def _subtree_role_count(value, target_roles: set[str]) -> int:
    count = 1 if _role(value) in target_roles else 0
    for child in _iter_structure_children(value):
        count += _subtree_role_count(child, target_roles)
    return count


def _subtree_has_meaningful_text(value) -> bool:
    resolved = _resolve(value)
    if isinstance(resolved, DictionaryObject):
        for key in MEANINGFUL_TEXT_KEYS:
            item = resolved.get(key)
            if item is not None and str(item).strip():
                return True
    return any(_subtree_has_meaningful_text(child) for child in _iter_structure_children(value))


def _has_content_reference(value, visited: set[tuple[int, int]] | None = None) -> bool:
    """Return True if a /K subtree contains page content or an object reference."""
    if visited is None:
        visited = set()

    if isinstance(value, IndirectObject):
        object_id = (value.idnum, value.generation)
        if object_id in visited:
            return False
        visited.add(object_id)
        value = value.get_object()

    if isinstance(value, (NumberObject, int)) and not isinstance(value, bool):
        # An integer in /K is a marked-content identifier (MCID).
        return True

    if isinstance(value, ArrayObject):
        return any(_has_content_reference(item, visited) for item in value)

    if isinstance(value, DictionaryObject):
        if value.get("/Type") in CONTENT_REFERENCE_TYPES:
            return True
        if "/MCID" in value or "/Obj" in value:
            return True
        if _is_structure_element(value):
            return "/K" in value and _has_content_reference(value["/K"], visited)
        # A non-structure dictionary in /K is unexpected and must be preserved.
        return True

    return False


def _is_prunable_empty_table_wrapper(value) -> bool:
    if _role(value) not in EMPTY_TABLE_WRAPPER_ROLES:
        return False
    roles = _subtree_roles(value)
    if not roles.issubset(ALLOWED_EMPTY_TABLE_SUBTREE_ROLES):
        return False
    if not ({"/TD", "/TH"} & roles):
        return False
    if _subtree_has_meaningful_text(value):
        return False
    return not _has_content_reference(value)


def _set_children(parent, original_children: list, kept_children: list) -> None:
    resolved = _resolve(parent)
    if not isinstance(resolved, DictionaryObject):
        raise RepairError("Structure-tree parent is not a dictionary.")

    if len(original_children) == len(kept_children):
        return
    if not kept_children:
        if "/K" in resolved:
            del resolved[NameObject("/K")]
    elif isinstance(resolved.get("/K"), ArrayObject) or len(kept_children) > 1:
        resolved[NameObject("/K")] = ArrayObject(kept_children)
    else:
        resolved[NameObject("/K")] = kept_children[0]


def repair_structure_tree(struct_tree_root: DictionaryObject) -> tuple[RepairStats, set[str]]:
    stats = RepairStats()
    removed_ids: set[str] = set()

    if "/K" not in struct_tree_root:
        raise RepairError("The PDF structure tree has no document root element.")

    def visit(value, parent_role: str = "", inside_toc_entry: bool = False) -> None:
        nonlocal stats, removed_ids
        resolved = _resolve(value)
        if not _is_structure_element(resolved):
            return

        role = _role(resolved)
        now_inside_toc = inside_toc_entry or role == "/TOCI"
        if now_inside_toc and role == "/Lbl":
            resolved[NameObject("/S")] = NameObject("/Span")
            stats.toc_labels_changed += 1
            role = "/Span"

        original_children = _children(resolved)
        kept_children = []
        for child in original_children:
            if (
                role == "/Table"
                and _is_structure_element(child)
                and _is_prunable_empty_table_wrapper(child)
            ):
                removed_ids.update(_subtree_ids(child))
                stats.table_wrappers_removed += 1
                stats.orphan_cells_removed += _subtree_role_count(child, {"/TD", "/TH"})
                continue

            kept_children.append(child)
            if _is_structure_element(child):
                visit(child, role, now_inside_toc)

        _set_children(resolved, original_children, kept_children)

    visit(struct_tree_root["/K"])
    return stats, removed_ids


def _prune_id_tree_node(node, removed_ids: set[str]) -> tuple[bool, str | None, str | None, int]:
    """Prune removed structure IDs from a PDF name tree."""
    resolved = _resolve(node)
    if not isinstance(resolved, DictionaryObject):
        raise RepairError("The structure ID tree contains a non-dictionary node.")

    removed_count = 0
    if "/Names" in resolved:
        names = resolved["/Names"]
        if not isinstance(names, ArrayObject) or len(names) % 2:
            raise RepairError("The structure ID tree contains a malformed /Names array.")
        kept = ArrayObject()
        for index in range(0, len(names), 2):
            name = str(names[index])
            if name in removed_ids:
                removed_count += 1
                continue
            kept.extend([names[index], names[index + 1]])

        if not kept:
            return False, None, None, removed_count

        resolved[NameObject("/Names")] = kept
        first = str(kept[0])
        last = str(kept[-2])
        resolved[NameObject("/Limits")] = ArrayObject([kept[0], kept[-2]])
        return True, first, last, removed_count

    if "/Kids" in resolved:
        kids = resolved["/Kids"]
        if not isinstance(kids, ArrayObject):
            raise RepairError("The structure ID tree contains a malformed /Kids array.")
        kept_kids = ArrayObject()
        first_limit = None
        last_limit = None
        for child in kids:
            keep, first, last, child_removed = _prune_id_tree_node(child, removed_ids)
            removed_count += child_removed
            if keep:
                kept_kids.append(child)
                first_limit = first_limit or first
                last_limit = last

        if not kept_kids:
            return False, None, None, removed_count

        resolved[NameObject("/Kids")] = kept_kids
        if first_limit is not None and last_limit is not None:
            limits = ArrayObject()
            # Preserve string-object encoding from the first and last child limits.
            first_child_limits = _resolve(kept_kids[0]).get("/Limits")
            last_child_limits = _resolve(kept_kids[-1]).get("/Limits")
            limits.extend([first_child_limits[0], last_child_limits[-1]])
            resolved[NameObject("/Limits")] = limits
        return True, first_limit, last_limit, removed_count

    raise RepairError("The structure ID tree node has neither /Names nor /Kids.")


def prune_structure_id_tree(struct_tree_root: DictionaryObject, removed_ids: set[str]) -> int:
    if not removed_ids:
        return 0
    id_tree_ref = struct_tree_root.get("/IDTree")
    if id_tree_ref is None:
        raise RepairError("Structure elements were removed, but the PDF has no /IDTree to update.")
    keep, _, _, removed_count = _prune_id_tree_node(id_tree_ref, removed_ids)
    if not keep:
        del struct_tree_root[NameObject("/IDTree")]
    if removed_count != len(removed_ids):
        missing = len(removed_ids) - removed_count
        raise RepairError(
            f"Removed {len(removed_ids)} structure elements but found only "
            f"{removed_count} matching ID-tree entries ({missing} missing)."
        )
    return removed_count


def validate_structure_tree(struct_tree_root: DictionaryObject) -> None:
    if "/K" not in struct_tree_root:
        raise RepairError("The repaired PDF structure tree has no document root element.")

    toc_labels: list[str] = []
    bad_cells: list[str] = []
    unsafe_wrappers: list[str] = []

    def visit(value, parent_role: str = "", inside_toc_entry: bool = False) -> None:
        if not _is_structure_element(value):
            return
        role = _role(value)
        now_inside_toc = inside_toc_entry or role == "/TOCI"
        if now_inside_toc and role == "/Lbl":
            toc_labels.append(role)
        if role in {"/TD", "/TH"} and parent_role != "/TR":
            bad_cells.append(f"{role} under {parent_role or 'no parent'}")
        if parent_role == "/Table" and role in EMPTY_TABLE_WRAPPER_ROLES:
            if _subtree_role_count(value, {"/TD", "/TH"}):
                unsafe_wrappers.append(role)
        for child in _iter_structure_children(value):
            visit(child, role, now_inside_toc)

    visit(struct_tree_root["/K"])

    failures = []
    if toc_labels:
        failures.append(f"{len(toc_labels)} TOC Lbl tag(s) remain")
    if bad_cells:
        failures.append(f"{len(bad_cells)} TH/TD tag(s) are not direct children of TR")
    if unsafe_wrappers:
        failures.append(f"{len(unsafe_wrappers)} table wrapper branch(es) still contain cells")
    if failures:
        raise RepairError("; ".join(failures) + ".")


def _get_struct_tree_root(pdf_root: DictionaryObject) -> DictionaryObject:
    struct_tree_ref = pdf_root.get("/StructTreeRoot")
    if struct_tree_ref is None:
        raise RepairError("The PDF is not tagged: /StructTreeRoot is missing.")
    struct_tree_root = _resolve(struct_tree_ref)
    if not isinstance(struct_tree_root, DictionaryObject):
        raise RepairError("The PDF /StructTreeRoot is malformed.")
    return struct_tree_root


def _validate_marked_pdf(pdf_root: DictionaryObject) -> None:
    mark_info = _resolve(pdf_root.get("/MarkInfo"))
    if not isinstance(mark_info, DictionaryObject) or not bool(mark_info.get("/Marked")):
        raise RepairError("The PDF is not marked as a tagged document.")


def repair_pdf(input_path: Path, output_path: Path) -> RepairStats:
    if input_path.resolve() == output_path.resolve():
        raise RepairError("Input and output paths must be different.")
    if not input_path.is_file():
        raise RepairError(f"Input PDF not found: {input_path}")

    reader = PdfReader(str(input_path), strict=True)
    original_page_count = len(reader.pages)
    writer = PdfWriter(clone_from=reader)
    writer.pdf_header = reader.pdf_header

    _validate_marked_pdf(writer.root_object)
    struct_tree_root = _get_struct_tree_root(writer.root_object)
    stats, removed_ids = repair_structure_tree(struct_tree_root)
    stats.id_tree_entries_removed = prune_structure_id_tree(struct_tree_root, removed_ids)
    validate_structure_tree(struct_tree_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        writer.write(output_file)

    repaired = PdfReader(str(output_path), strict=True)
    if len(repaired.pages) != original_page_count:
        raise RepairError(
            f"Page count changed from {original_page_count} to {len(repaired.pages)} during repair."
        )
    _validate_marked_pdf(repaired.trailer["/Root"])
    validate_structure_tree(_get_struct_tree_root(repaired.trailer["/Root"]))
    return stats


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair narrowly scoped table and TOC tag artifacts in a tagged PDF."
    )
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        stats = repair_pdf(args.input_pdf, args.output_pdf)
    except Exception as exc:
        print(f"Error repairing PDF tags: {exc}", file=sys.stderr)
        return 1

    print(
        "PDF tag repair complete: "
        f"converted {stats.toc_labels_changed} TOC Lbl tag(s) to Span; "
        f"removed {stats.table_wrappers_removed} empty table wrapper branch(es) "
        f"containing {stats.orphan_cells_removed} orphan TH/TD tag(s); "
        f"removed {stats.id_tree_entries_removed} associated ID-tree entry/entries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
