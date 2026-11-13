//! The constructs and the named failures of the template subset, one
//! case each. The outputs and the failures of the reference cases came
//! from the renderer of transformers 5.5.2 with jinja2 3.1.6 on
//! 2026-11-14, with the variables of VARIABLES and the date 2026-01-05.

use super::{Date, ErrorKind, Origin, Template, Value, json};
use crate::prelude::*;

/// The variables of every case
const VARIABLES: &str = r#"{"d": {"b": 2, "a": 1, "C": 3}, "items": [{"name": "x", "type": "f", "n": 3}, {"name": "y", "type": "g", "n": 1}], "s": "  Hello, World  ", "n": null, "words": ["b", "A", "c"], "t": true}"#;

const DATE: Date = Date {
    year: 2026,
    month: 1,
    day: 5,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Raised,
    Type,
    Undefined,
}

enum Expect {
    Renders(&'static str),
    Fails(Kind),
}

/// The cases of the reference: a name, a template and its outcome
const REFERENCE_CASES: [(&str, &str, Expect); 52] = [
    (
        r#"if"#,
        r#"{% if 1 > 2 %}a{% elif t %}b{% else %}c{% endif %}"#,
        Expect::Renders(r#"b"#),
    ),
    (
        r#"loop_vars"#,
        r#"{% for x in words %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }}{{ loop.revindex }}|{% endfor %}"#,
        Expect::Renders(r#"10TrueFalse33|21FalseFalse32|32FalseTrue31|"#),
    ),
    (
        r#"loop_prev_next"#,
        r#"{% for x in words %}[{{ loop.previtem }}-{{ loop.nextitem }}]{% endfor %}"#,
        Expect::Renders(r#"[-A][b-c][A-]"#),
    ),
    (
        r#"for_else"#,
        r#"{% for x in [] %}a{% else %}empty{% endfor %}"#,
        Expect::Renders(r#"empty"#),
    ),
    (
        r#"for_filter"#,
        r#"{% for x in [1,2,3,4] if x % 2 == 0 %}{{ x }}{% endfor %}"#,
        Expect::Renders(r#"24"#),
    ),
    (
        r#"for_items"#,
        r#"{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}"#,
        Expect::Renders(r#"b=2;a=1;C=3;"#),
    ),
    (
        r#"break_continue"#,
        r#"{% for x in [1,2,3,4,5] %}{% if x == 2 %}{% continue %}{% endif %}{% if x == 4 %}{% break %}{% endif %}{{ x }}{% endfor %}"#,
        Expect::Renders(r#"13"#),
    ),
    (
        r#"loop_scope"#,
        r#"{% set x = 0 %}{% for i in [1,2] %}{% set x = i %}{{ x }}{% endfor %}{{ x }}"#,
        Expect::Renders(r#"120"#),
    ),
    (
        r#"namespace"#,
        r#"{% set ns = namespace(c=0) %}{% for i in [1,2,3] %}{% set ns.c = ns.c + i %}{% endfor %}{{ ns.c }}"#,
        Expect::Renders(r#"6"#),
    ),
    (
        r#"set_tuple"#,
        r#"{% set a, b = [1, 2] %}{{ b }}{{ a }}"#,
        Expect::Renders(r#"21"#),
    ),
    (
        r#"set_block"#,
        r#"{% set x %}in {{ 1 + 1 }}{% endset %}[{{ x }}]"#,
        Expect::Renders(r#"[in 2]"#),
    ),
    (
        r#"macro"#,
        r#"{% macro m(a, b='B') %}{{ a }}-{{ b }}-{{ t }}{% endmacro %}{{ m(1) }} {{ m(2, b='c') }}"#,
        Expect::Renders(r#"1-B-True 2-c-True"#),
    ),
    (
        r#"whitespace"#,
        r#"a
  {% if true %}
  b {{- ' c' -}}  
  {% endif %}
d
"#,
        Expect::Renders(
            r#"a
  b cd"#,
        ),
    ),
    (
        r#"comment"#,
        r#"a {# note #}
b {#- x -#} c"#,
        Expect::Renders(r#"a bc"#),
    ),
    (
        r#"escapes"#,
        r#"{{ 'a\tb\n\x41\u00e9\\' }}"#,
        Expect::Renders(
            r#"a	b
Aé\"#,
        ),
    ),
    (
        r#"slices"#,
        r#"{{ words[::-1] }}{{ s[2:7] }}{{ words[-1] }}{{ words[1:] }}"#,
        Expect::Renders(r#"['c', 'A', 'b']Helloc['A', 'c']"#),
    ),
    (
        r#"compare_chain"#,
        r#"{{ 1 < 2 < 3 }}{{ 3 > 2 > 2 }}{{ 'a' in 'cat' }}{{ 4 not in [1,2] }}"#,
        Expect::Renders(r#"TrueFalseTrueTrue"#),
    ),
    (
        r#"and_or"#,
        r#"{{ 0 or 'x' }}{{ 'y' and 'z' }}{{ n or 'none' }}"#,
        Expect::Renders(r#"xznone"#),
    ),
    (
        r#"cond_expr"#,
        r#"{{ 'a' if t else 'b' }}{{ 'c' if n }}."#,
        Expect::Renders(r#"a."#),
    ),
    (
        r#"arith"#,
        r#"{{ 7 // 2 }} {{ -7 // 2 }} {{ -7 % 3 }} {{ 2 ** 10 }} {{ 7 / 2 }} {{ 1 + 2.5 }} {{ 'ab' * 2 }}"#,
        Expect::Renders(r#"3 -4 2 1024 3.5 3.5 abab"#),
    ),
    (
        r#"concat"#,
        r#"{{ 1 ~ 'a' ~ n ~ t }}"#,
        Expect::Renders(r#"1aNoneTrue"#),
    ),
    (
        r#"output_forms"#,
        r#"{{ 1.0 }} {{ 1e20 }} {{ 0.1 + 0.2 }} {{ [1, 'a', none, true] }} {{ {'k': 1} }} {{ (1,) }} {{ none }}"#,
        Expect::Renders(r#"1.0 1e+20 0.30000000000000004 [1, 'a', None, True] {'k': 1} (1,) None"#),
    ),
    (
        r#"default"#,
        r#"{{ n | default('dn') }}{{ n | default('db', true) }}{{ d.zz | default('dz') }}"#,
        Expect::Renders(r#"Nonedbdz"#),
    ),
    (
        r#"length_string"#,
        r#"{{ words | length }}{{ d | count }}{{ 12 | string }}{{ s | length }}"#,
        Expect::Renders(r#"331216"#),
    ),
    (
        r#"trim"#,
        r#"[{{ s | trim }}][{{ 'xxaxx' | trim('x') }}]"#,
        Expect::Renders(r#"[Hello, World][a]"#),
    ),
    (
        r#"items_filter"#,
        r#"{% for k, v in d | items %}{{ k }}{{ v }}{% endfor %}"#,
        Expect::Renders(r#"b2a1C3"#),
    ),
    (
        r#"safe_markup"#,
        r#"{{ 'a'|safe + '<"&\'>' }}{{ '<b>' + '<i>'|safe }}{{ '<'|safe ~ '>' }}"#,
        Expect::Renders(r#"a&lt;&#34;&amp;&#39;&gt;&lt;b&gt;<i><>"#),
    ),
    (
        r#"join"#,
        r#"{{ words | join(', ') }}|{{ items | join('-', attribute='name') }}"#,
        Expect::Renders(r#"b, A, c|x-y"#),
    ),
    (
        r#"list_first_last"#,
        r#"{{ 'abc' | list }}{{ words | first }}{{ words | last }}"#,
        Expect::Renders(r#"['a', 'b', 'c']bc"#),
    ),
    (
        r#"select"#,
        r#"{{ items | selectattr('type', 'equalto', 'f') | map(attribute='name') | list }}{{ items | rejectattr('n', 'gt', 2) | map(attribute='name') | join }}{{ [1,2,3,4] | select('odd') | list }}{{ [1,2,3] | reject('equalto', 2) | list }}"#,
        Expect::Renders(r#"['x']y[1, 3][1, 3]"#),
    ),
    (
        r#"map_filter"#,
        r#"{{ words | map('upper') | join }}"#,
        Expect::Renders(r#"BAC"#),
    ),
    (
        r#"dictsort"#,
        r#"{{ d | dictsort }}{{ d | dictsort(true) }}"#,
        Expect::Renders(r#"[('a', 1), ('b', 2), ('C', 3)][('C', 3), ('a', 1), ('b', 2)]"#),
    ),
    (
        r#"sort"#,
        r#"{{ words | sort }}{{ words | sort(reverse=true, case_sensitive=true) }}{{ items | sort(attribute='n') | map(attribute='name') | join }}"#,
        Expect::Renders(r#"['A', 'b', 'c']['c', 'b', 'A']yx"#),
    ),
    (
        r#"case"#,
        r#"{{ 'aB' | lower }}{{ 'aB' | upper }}{{ 'aB'.title() }}"#,
        Expect::Renders(r#"abABAb"#),
    ),
    (
        r#"int"#,
        r#"{{ '42' | int }}{{ '4.7' | int }}{{ 'x' | int }}{{ 3.9 | int }}"#,
        Expect::Renders(r#"42403"#),
    ),
    (
        r#"replace"#,
        r#"{{ 'aaa' | replace('a', 'b', 2) }}{{ 'a.b'.replace('.', '-') }}"#,
        Expect::Renders(r#"bbaa-b"#),
    ),
    (
        r#"tojson"#,
        r#"{{ d | tojson }}|{{ items[0] | tojson(indent=2) }}|{{ 'é"' | tojson }}|{{ 'é' | tojson(ensure_ascii=true) }}|{{ [] | tojson(indent=4) }}|{{ 1.5 | tojson }}"#,
        Expect::Renders(
            r#"{"b": 2, "a": 1, "C": 3}|{
  "name": "x",
  "type": "f",
  "n": 3
}|"é\""|"\u00e9"|[]|1.5"#,
        ),
    ),
    (
        r#"tests"#,
        r#"{{ x is defined }}{{ x is undefined }}{{ n is none }}{{ s is string }}{{ d is mapping }}{{ words is iterable }}{{ words is sequence }}{{ 1 is number }}{{ 1 is integer }}{{ 1.0 is float }}{{ t is boolean }}{{ t is true }}{{ t is false }}"#,
        Expect::Renders(r#"FalseTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueTrueFalse"#),
    ),
    (
        r#"tests_args"#,
        r#"{{ 2 is equalto 2 }}{{ n is sameas none }}{{ 'a' is in ['a'] }}{{ 3 is odd }}{{ 4 is even }}{{ 9 is divisibleby 3 }}{{ 'ab' is lower }}{{ 'AB' is upper }}{{ 'trim' is filter }}{{ 'from_json' is filter }}{{ 'odd' is test }}"#,
        Expect::Renders(r#"TrueTrueTrueTrueTrueTrueTrueTrueTrueFalseTrue"#),
    ),
    (
        r#"split"#,
        r#"{{ ' a  b c '.split() }}{{ 'a,b,,c'.split(',') }}{{ 'a b c'.split(' ', 1) }}{{ 'a b c'.rsplit(' ', 1) }}"#,
        Expect::Renders(r#"['a', 'b', 'c']['a', 'b', '', 'c']['a', 'b c']['a b', 'c']"#),
    ),
    (
        r#"strip_methods"#,
        r#"[{{ s.strip() }}][{{ s.lstrip() }}][{{ s.rstrip() }}][{{ 'xax'.strip('x') }}]"#,
        Expect::Renders(r#"[Hello, World][Hello, World  ][  Hello, World][a]"#),
    ),
    (
        r#"starts_ends"#,
        r#"{{ 'abc'.startswith('ab') }}{{ 'abc'.startswith(('x', 'a')) }}{{ 'abc'.endswith('bc') }}{{ 'abc'.find('c') }}{{ 'aXaX'.count('X') }}"#,
        Expect::Renders(r#"TrueTrueTrue22"#),
    ),
    (
        r#"dict_methods"#,
        r#"{{ d.get('a') }}{{ d.get('zz', 'no') }}{{ d.get('zz') }}{{ d.keys() | list }}{{ d.values() | list }}"#,
        Expect::Renders(r#"1noNone['b', 'a', 'C'][2, 1, 3]"#),
    ),
    (
        r#"range"#,
        r#"{{ range(3) | list }}{{ range(1, 7, 2) | list }}"#,
        Expect::Renders(r#"[0, 1, 2][1, 3, 5]"#),
    ),
    (
        r#"strftime"#,
        r#"{{ strftime_now('%Y-%m-%d') }} {{ strftime_now('%d %b %Y') }} {{ strftime_now('%B %d, %Y') }}"#,
        Expect::Renders(r#"2026-01-05 05 Jan 2026 January 05, 2026"#),
    ),
    (
        r#"attribute_undefined"#,
        r#"[{{ d.zz }}][{{ items[5] }}][{{ d['zz'] }}][{{ d.zz | length }}]"#,
        Expect::Renders(r#"[][][][0]"#),
    ),
    (
        r#"undefined_eq"#,
        r#"{{ d.zz == d.yy }}{{ d.zz == 1 }}{{ d.zz is defined }}"#,
        Expect::Renders(r#"TrueFalseFalse"#),
    ),
    (
        r#"in_unset"#,
        r#"{{ 'a' in unset_list }}{% for x in unset_list %}x{% endfor %}{% if unset_flag %}y{% endif %}{{ unset_flag is defined }}"#,
        Expect::Renders(r#"FalseFalse"#),
    ),
    (
        r#"raise"#,
        r#"{{ raise_exception('stop here') }}"#,
        Expect::Fails(Kind::Raised),
    ),
    (
        r#"type_iter_none"#,
        r#"{% for x in n %}{% endfor %}"#,
        Expect::Fails(Kind::Type),
    ),
    (r#"type_add"#, r#"{{ 'a' + 1 }}"#, Expect::Fails(Kind::Type)),
    (
        r#"undefined_attr"#,
        r#"{{ d.zz.yy }}"#,
        Expect::Fails(Kind::Undefined),
    ),
];

fn variables() -> Vec<(String, Value)> {
    let Value::Map(m) = json::parse(VARIABLES, Origin::Content).expect("the variables") else {
        panic!("the variables are a mapping");
    };
    m.iter()
        .map(|(k, v)| match k {
            Value::Str(k) => (k.as_str().into(), v.clone()),
            _ => panic!("a key that is not a string"),
        })
        .collect()
}

/// The names that the cases set
fn names(vars: &[(String, Value)]) -> Vec<&str> {
    vars.iter().map(|(n, _)| n.as_str()).collect()
}

fn render(source: &str, date: Option<Date>) -> Result<String, super::TemplateError> {
    let vars = variables();
    let t = Template::load(source, &names(&vars))?;
    let segments = t.render(&vars, date)?;
    Ok(segments.iter().map(|s| s.text.as_str()).collect())
}

fn kind_of(e: &ErrorKind) -> Option<Kind> {
    match e {
        ErrorKind::Raised(_) => Some(Kind::Raised),
        ErrorKind::Type(_) => Some(Kind::Type),
        ErrorKind::Undefined(_) => Some(Kind::Undefined),
        _ => None,
    }
}

#[test]
fn each_construct_renders_as_the_reference() {
    for (name, source, expect) in REFERENCE_CASES {
        match (render(source, Some(DATE)), expect) {
            (Ok(out), Expect::Renders(want)) => assert_eq!(out, want, "{name}"),
            (Err(e), Expect::Fails(kind)) => {
                assert_eq!(kind_of(&e.kind), Some(kind), "{name}: {e}")
            }
            (Ok(out), Expect::Fails(kind)) => {
                panic!("{name}: renders {out:?}, the reference fails with {kind:?}")
            }
            (Err(e), Expect::Renders(want)) => {
                panic!("{name}: {e}, the reference renders {want:?}")
            }
        }
    }
}

/// The failures of biv only: an outcome that the subset or the rulings give
#[test]
fn each_failure_of_the_subset_has_its_name() {
    let fails = |source: &str, date: Option<Date>| render(source, date).expect_err(source).kind;
    // an unset top-level name used as a value fails at load
    assert_eq!(
        fails("{{ unset_name }}", Some(DATE)),
        ErrorKind::VariableUnknown("unset_name".into())
    );
    assert_eq!(
        fails("{{ unset_x.y if false else '' }}", Some(DATE)),
        ErrorKind::VariableUnknown("unset_x".into())
    );
    // a name that the template sets, read before the set, fails at render
    assert_eq!(
        fails(
            "{% if false %}{% set late = 1 %}{% endif %}{{ late }}",
            Some(DATE)
        ),
        ErrorKind::VariableUnknown("late".into())
    );
    // a filter that the reference does not have loads, is listed, and fails
    // when a render reaches it
    let vars = variables();
    let t = Template::load("{% if t %}{{ 1 | from_json }}{% endif %}", &names(&vars))
        .expect("it loads");
    assert_eq!(
        t.report().unknown_filters,
        vec![("from_json".to_string(), 1)]
    );
    assert_eq!(
        t.render(&vars, Some(DATE)).expect_err("it is reached").kind,
        ErrorKind::FilterUnknown("from_json".into())
    );
    let unreached = Template::load("{% if n %}{{ 1 | from_json }}{% endif %}ok", &names(&vars))
        .expect("it loads");
    let out: String = unreached
        .render(&vars, Some(DATE))
        .expect("not reached")
        .iter()
        .map(|s| s.text.as_str())
        .collect();
    assert_eq!(out, "ok");
    // the date
    assert_eq!(
        fails("{{ strftime_now('%Y') }}", None),
        ErrorKind::DateMissing
    );
    assert_eq!(
        fails("{{ strftime_now('%H') }}", Some(DATE)),
        ErrorKind::DateFormat('H')
    );
    // the constructs that the subset leaves out fail at load
    for (source, what) in [
        ("{% filter upper %}a{% endfilter %}", "the filter block"),
        ("{{ [1].append(2) }}", "the list method append"),
        ("{{ 'a' | indent }}", "indent"),
        ("{{ 'a' | unique }}", "unique"),
        ("{{ 'a'.format() }}", "a method outside the subset"),
        ("{% include 'x' %}", "include"),
    ] {
        assert_eq!(
            fails(source, Some(DATE)),
            ErrorKind::Unsupported(what),
            "{source}"
        );
    }
    // text that is not a template
    assert!(matches!(
        fails("{% if %}", Some(DATE)),
        ErrorKind::Syntax(_)
    ));
    assert!(matches!(
        fails("{{ 1 + }}", Some(DATE)),
        ErrorKind::Syntax(_)
    ));
    assert!(matches!(
        fails("{% for x in y %}", Some(DATE)),
        ErrorKind::Syntax(_)
    ));
}

/// The options of a template: the names that it reads only under a guard
/// or as a test
#[test]
fn the_options_of_a_template_are_its_guarded_reads() {
    let vars = variables();
    let t = Template::load(
        "{% if opt_a is defined %}{{ opt_a }}{% endif %}{{ opt_b | default('x') }}{% for i in opt_c %}{% endfor %}{{ 'k' in opt_d }}",
        &names(&vars),
    )
    .expect("it loads");
    assert_eq!(t.report().options, vec!["opt_a", "opt_b", "opt_c", "opt_d"]);
}

/// The origin of each byte: the text of a value of the caller stays
/// Content through slices, trims, splits, case changes and joins
#[test]
fn the_text_of_a_value_keeps_its_origin() {
    let vars = vec![("m".to_string(), Value::str(" <|x|> ab ", Origin::Content))];
    let t = Template::load(
        "<|s|>{{ m | trim }}|{{ m.split() | join(',') }}|{{ m[1:5] | upper }}<|e|>",
        &["m"],
    )
    .expect("it loads");
    let segments = t.render(&vars, None).expect("it renders");
    let of = |o: Origin| -> String {
        segments
            .iter()
            .filter(|s| s.origin == o)
            .map(|s| s.text.as_str())
            .collect()
    };
    assert_eq!(of(Origin::Template), "<|s|>|,|<|e|>");
    assert_eq!(of(Origin::Content), "<|x|> ab<|x|>ab<|X|");
}
