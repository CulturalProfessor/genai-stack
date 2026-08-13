import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack import extract as fs

TS_SOURCE = """\
import { Accounts } from 'meteor/accounts-base';

export interface CrowdUser {
\tcrowd_username: string;
}

type Handler = (id: string) => Promise<void>;

export class CrowdSync {
\tprivate readonly url: string;

\tconstructor(url: string) {
\t\tthis.url = url;
\t}

\tasync syncDataToUser(crowdUser: Record<string, any>, id: string) {
\t\tconst user = { username: crowdUser.username };
\t\treturn user;
\t}
}

export const setReaction = async (room: string) => {
\tconst oldMessage = JSON.parse(room);
\treturn oldMessage;
};

function cleanUsername(username: string) {
\treturn username.trim();
}
"""


def symbols_by_name(source: str = TS_SOURCE, path: str = "src/crowd.ts", **kwargs):
    symbols = fs.extract_symbols(source.encode(), "typescript", path, **kwargs)
    return {symbol.qualified_name: symbol for symbol in symbols}


def test_extracts_every_declaration_kind():
    found = symbols_by_name()
    assert set(found) == {
        "CrowdUser",
        "Handler",
        "CrowdSync",
        "CrowdSync.constructor",
        "CrowdSync.syncDataToUser",
        "setReaction",
        "cleanUsername",
    }


def test_kinds_are_labelled():
    found = symbols_by_name()
    assert found["CrowdUser"].kind == "interface"
    assert found["Handler"].kind == "type_alias"
    assert found["CrowdSync"].kind == "class"
    assert found["CrowdSync.syncDataToUser"].kind == "method"
    assert found["setReaction"].kind == "function"
    assert found["cleanUsername"].kind == "function"


def test_exported_declarations_are_not_skipped():
    """The previous line-based extractor dropped any line containing `export`."""
    found = symbols_by_name()
    assert "setReaction" in found
    assert "CrowdSync" in found


def test_code_has_no_duplicated_lines():
    """The old extractor appended each opening line twice."""
    code = symbols_by_name()["CrowdSync.syncDataToUser"].code
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    assert len(lines) == len(set(lines))


def test_method_body_is_complete():
    code = symbols_by_name()["CrowdSync.syncDataToUser"].code
    assert code.startswith("async syncDataToUser(")
    assert code.rstrip().endswith("}")
    assert "return user;" in code


def test_class_skeleton_lists_members_without_bodies():
    code = symbols_by_name()["CrowdSync"].code
    assert "class CrowdSync {" in code
    assert "async syncDataToUser(crowdUser: Record<string, any>, id: string)" in code
    # Member bodies belong to the method symbols, not to the class symbol.
    assert "return user;" not in code


def test_arrow_function_signature_stops_at_the_body():
    found = symbols_by_name()
    assert found["setReaction"].signature == "setReaction = async (room: string) =>"


def test_line_numbers_point_at_the_declaration():
    symbol = symbols_by_name()["cleanUsername"]
    assert TS_SOURCE.splitlines()[symbol.start_line - 1].startswith("function cleanUsername")


def test_ids_are_unique_and_carry_the_file_path():
    symbols = fs.extract_symbols(TS_SOURCE.encode(), "typescript", "src/crowd.ts")
    ids = [symbol.id for symbol in symbols]
    assert len(ids) == len(set(ids))
    assert all(symbol.id.startswith("src/crowd.ts:") for symbol in symbols)


def test_same_name_in_two_files_yields_different_ids():
    """Symbol names repeat across a codebase; ids must not collide."""
    a = fs.extract_symbols(TS_SOURCE.encode(), "typescript", "a.ts")
    b = fs.extract_symbols(TS_SOURCE.encode(), "typescript", "b.ts")
    assert not {s.id for s in a} & {s.id for s in b}


def test_truncation_is_flagged():
    found = symbols_by_name(max_chars=40)
    symbol = found["CrowdSync.syncDataToUser"]
    assert symbol.truncated
    assert symbol.code.endswith("// ... truncated")


def test_min_chars_drops_trivial_symbols():
    found = symbols_by_name(min_chars=10_000)
    assert found == {}


def test_javascript_and_tsx_are_supported():
    js = fs.extract_symbols(b"export const go = () => { return 1; }", "javascript", "a.js")
    tsx = fs.extract_symbols(
        b"export const View = (p: Props) => { return <div />; }", "tsx", "a.tsx"
    )
    assert [s.name for s in js] == ["go"]
    assert [s.name for s in tsx] == ["View"]


def test_unparseable_source_does_not_raise():
    symbols = fs.extract_symbols(b"function ( { { {", "typescript", "broken.ts")
    assert isinstance(symbols, list)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "crowd.ts").write_text(TS_SOURCE)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("export const dep = () => 1;")
    (tmp_path / "src" / "crowd.test.ts").write_text(TS_SOURCE)
    (tmp_path / "src" / "readme.md").write_text("# not code")
    return tmp_path


def test_build_index_prunes_ignored_paths(repo):
    records = fs.build_index(
        repo, set(fs.DEFAULT_IGNORE_DIRS), fs.DEFAULT_IGNORE_GLOBS, 4000, 0
    )
    paths = [record.path for record in records]
    assert paths == ["src/crowd.ts"]


def test_main_writes_a_loadable_index(repo, tmp_path):
    output = tmp_path / "out" / "index.json"
    assert fs.main(["--repo", str(repo), "--output", str(output)]) == 0

    index = json.loads(output.read_text())
    assert index["files"][0]["language"] == "typescript"
    assert index["files"][0]["depth"] == 1
    assert any(s["name"] == "cleanUsername" for s in index["files"][0]["symbols"])


def test_main_reports_a_missing_repo(tmp_path):
    assert fs.main(["--repo", str(tmp_path / "nope")]) == 1


GRAPH_SOURCE = """\
import { cleanUsername } from './util/text';
import { Accounts } from 'meteor/accounts-base';
const legacy = require('../legacy/sync');

export class CrowdSync {
\tasync syncDataToUser(user: any, id: string) {
\t\tconst username = cleanUsername(user.name);
\t\tawait Accounts.setPasswordAsync(id, user.password);
\t\treturn username;
\t}
}

export const runSync = async (user: any) => {
\tconst sync = new CrowdSync();
\treturn sync.syncDataToUser(user, user.id);
};
"""


def test_calls_are_collected_per_symbol():
    found = symbols_by_name(GRAPH_SOURCE)
    assert found["CrowdSync.syncDataToUser"].calls == [
        "cleanUsername",
        "setPasswordAsync",
    ]


def test_new_expressions_are_recorded_apart_from_calls():
    """What code builds and what it invokes are different questions."""
    found = symbols_by_name(GRAPH_SOURCE)
    assert "CrowdSync" in found["runSync"].instantiates
    assert "CrowdSync" not in found["runSync"].calls


def test_method_calls_use_the_property_name():
    found = symbols_by_name(GRAPH_SOURCE)
    assert "syncDataToUser" in found["runSync"].calls


def test_class_skeleton_has_no_calls_of_its_own():
    """Calls inside a body belong to that method, not to the class symbol."""
    assert symbols_by_name(GRAPH_SOURCE)["CrowdSync"].calls == []


def test_recursive_calls_do_not_self_link():
    source = "function walk(n: any) { return walk(n.next); }"
    assert symbols_by_name(source)["walk"].calls == []


def test_imports_cover_import_and_require():
    _, imports, _ = fs.parse_file(GRAPH_SOURCE.encode(), "typescript", "src/crowd.ts")
    assert imports == ["../legacy/sync", "./util/text", "meteor/accounts-base"]


def test_build_index_records_imports(tmp_path):
    (tmp_path / "a.ts").write_text("import { b } from './b';\nexport function a() { return b(); }")
    (tmp_path / "b.ts").write_text("export function b() { return 1; }")
    records = {
        r.path: r
        for r in fs.build_index(tmp_path, set(fs.DEFAULT_IGNORE_DIRS), [], 4000, 0)
    }
    assert records["a.ts"].imports == ["./b"]
    assert records["b.ts"].imports == []


TYPE_SOURCE = """\
export interface RetryOptions {
\t/**
\t * The number of times to retry failed requests.
\t * Long explanation that dominates the embedding.
\t */
\tlimit?: number;
\t/** Methods to retry. */
\tmethods?: string[];
}

export type Handler = (id: string) => Promise<void>;

export enum Color {
\tRed = 1,
\tBlue = 2,
}
"""


def test_interface_keeps_members_and_drops_doc_comments():
    code = symbols_by_name(TYPE_SOURCE)["RetryOptions"].code
    assert code.startswith("interface RetryOptions {")
    # The trailing `;` is a separate token, not part of the member node.
    assert "limit?: number" in code
    assert "methods?: string[]" in code
    # The JSDoc is what was drowning out implementation code in search.
    assert "dominates the embedding" not in code
    assert "Methods to retry" not in code


def test_enum_keeps_its_members():
    code = symbols_by_name(TYPE_SOURCE)["Color"].code
    assert code.startswith("enum Color {")
    assert "Red = 1" in code and "Blue = 2" in code


def test_short_type_alias_is_kept_whole():
    code = symbols_by_name(TYPE_SOURCE)["Handler"].code
    assert code == "type Handler = (id: string) => Promise<void>;"


def test_object_type_alias_becomes_a_skeleton():
    source = "type Shape = {\n\t/** docs */\n\ta: number;\n\tb: string;\n};"
    code = symbols_by_name(source)["Shape"].code
    assert code.startswith("type Shape = {")
    assert "a: number" in code and "docs" not in code


def test_type_skeleton_caps_very_long_member_lists():
    members = "\n".join(f"\tf{i}: number;" for i in range(60))
    code = symbols_by_name(f"interface Big {{\n{members}\n}}")["Big"].code
    assert "20 more members" in code  # 60 members, capped at 40


def test_types_have_no_call_edges():
    assert symbols_by_name(TYPE_SOURCE)["RetryOptions"].calls == []


DOC_SOURCE = """
/* eslint-disable no-console */

/** Resolves the sidebar path for a doc. */
export function resolveSidebar(a: string): Sidebar {
    return load(a);
}

// picks the first match
// and falls back to null
const findFirst = (xs: string[]) => xs[0];

/** Not this one. */

export function detached(): void {}

class Widget {
    /** Renders the widget. */
    render(): string { return "x"; }

    other(): string { return "y"; }
}
"""


def docs():
    symbols = fs.extract_symbols(DOC_SOURCE.encode(), "typescript", "a.ts")
    return {symbol.name: symbol.doc for symbol in symbols}


def test_jsdoc_above_an_exported_function_is_captured():
    """The comment is a sibling of the export_statement, not of the
    function_declaration that gets recorded."""
    assert docs()["resolveSidebar"] == "/** Resolves the sidebar path for a doc. */"


def test_consecutive_line_comments_are_captured_together():
    assert docs()["findFirst"] == "// picks the first match\n// and falls back to null"


def test_a_comment_separated_by_a_blank_line_is_not_attached():
    """Otherwise every file-header banner lands on the first declaration."""
    assert docs()["detached"] is None


def test_a_method_takes_the_comment_above_it():
    assert docs()["render"] == "/** Renders the widget. */"


def test_a_symbol_without_a_comment_has_no_doc():
    assert docs()["other"] is None


def test_the_doc_is_kept_out_of_the_code():
    symbols = fs.extract_symbols(DOC_SOURCE.encode(), "typescript", "a.ts")
    resolve = next(s for s in symbols if s.name == "resolveSidebar")
    assert "Resolves the sidebar path" not in resolve.code


def test_an_overlong_comment_is_dropped():
    source = f"/* {'x' * 2000} */\nexport function big(): void {{}}".encode()
    assert fs.extract_symbols(source, "typescript", "a.ts")[0].doc is None


INTERFACE_SOURCE = """
import {resolve, join as joinPath} from './paths';
import loadConfig from './config';
import * as helpers from './helpers';

export {parse} from './parser';
export * from './types';

export function build(): void {}
export const limit = 5;
const internal = () => {};
export {internal};
"""


def interface():
    return fs.extract_interface(INTERFACE_SOURCE.encode(), "typescript")


def binding(local):
    return next(b for b in interface().bindings if b["local"] == local)


def test_a_named_import_records_where_the_name_came_from():
    assert binding("resolve") == {
        "local": "resolve", "imported": "resolve", "specifier": "./paths"
    }


def test_an_aliased_import_keeps_both_names():
    """The call site uses the alias; the source module knows the original."""
    assert binding("joinPath")["imported"] == "join"


def test_a_default_import_is_marked_as_such():
    assert binding("loadConfig")["imported"] == "default"


def test_a_namespace_import_is_recorded():
    assert binding("helpers")["imported"] == "*"


def test_a_re_export_names_the_module_it_passes_through_from():
    assert {"specifier": "./parser", "imported": "parse", "local": "parse"} in (
        interface().re_exports
    )


def test_a_star_re_export_is_recorded_so_a_barrel_can_be_followed():
    assert {"specifier": "./types", "imported": "*", "local": "*"} in (
        interface().re_exports
    )


def test_exported_names_cover_declarations_and_export_clauses():
    names = set(interface().exported_names)
    assert {"build", "limit", "internal"} <= names


def test_symbols_are_flagged_as_exported():
    symbols = {s.name: s for s in fs.extract_symbols(INTERFACE_SOURCE.encode(), "typescript", "a.ts")}
    assert symbols["build"].exported is True
    assert symbols["internal"].exported is True


def test_a_symbol_that_is_not_exported_is_not_flagged():
    symbols = fs.extract_symbols(b"function hidden(): void {}", "typescript", "a.ts")
    assert symbols[0].exported is False


HERITAGE_SOURCE = """
interface Options { retries: number }

export class Client extends BaseClient implements Closeable {
    connect(options: Options): Session { return open(options); }
}

export function widen(input: Options): Session { const x: Local = 1; return x; }
"""


def test_a_class_records_what_it_extends_and_implements():
    found = symbols_by_name(HERITAGE_SOURCE)
    assert found["Client"].extends == ["BaseClient"]
    assert found["Client"].implements == ["Closeable"]


def test_a_signature_records_its_parameter_and_return_types():
    assert symbols_by_name(HERITAGE_SOURCE)["widen"].type_names == ["Options", "Session"]


def test_types_used_only_inside_the_body_are_not_recorded():
    """The body's annotations describe the implementation, not the interface."""
    assert "Local" not in symbols_by_name(HERITAGE_SOURCE)["widen"].type_names


def test_a_method_records_its_own_signature_types():
    found = symbols_by_name(HERITAGE_SOURCE)
    assert found["Client.connect"].type_names == ["Options", "Session"]


def test_an_index_records_the_commit_it_was_built_from(tmp_path):
    """Gold names files and symbols at a point in history. An index that
    cannot say which commit it came from cannot tell rotted gold from a
    retrieval regression."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.ts").write_text("export function f() {}\n")
    for command in (
        ["git", "-C", str(tmp_path), "add", "."],
        ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "first"],
    ):
        subprocess.run(command, check=True)

    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert fs.git_commit(tmp_path) == head


def test_a_directory_that_is_not_a_checkout_records_no_commit(tmp_path):
    assert fs.git_commit(tmp_path) is None
