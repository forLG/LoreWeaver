# Data Overview

DnD 模组以 json 格式存储，顶层 array 名为 `data`，每个元素为字典，包含各章节的内容。 每个章节字典包含以下字段：
- type: 该字典的类型
- name: 该字典的名称
- page: 该章节在原始书籍中的页码
- id: 该字典的唯一标识符
- entries: 该字典的内容，通常为一个列表，包含文本段落、子章节等。

## 支持的 type 类型

共有 13 种 type 类型，具体如下表所示：

| Type           | Count | Purpose                                     | Key Properties                     |
  |----------------|-------|---------------------------------------------|------------------------------------|
  | section        | 31    | Major content chapters/sections             | id, name, page, entries            |
  | entries        | 155   | Named content containers (subsections)      | id, name, page, entries            |
  | inset          | 4     | Sidebar/inset boxes                         | id, name, page, entries            |
  | insetReadaloud | 52    | Boxed text to read aloud to players         | id, name, page, entries            |
  | list           | 23    | Bullet/numbered lists                       | items, style, columns              |
  | item           | 44    | List items (inside lists)                   | name, entry, entries               |
  | table          | 4     | Data tables                                 | rows, colLabels, caption           |
  | gallery        | 5     | Image collections (e.g., map variations)    | images                             |
  | image          | 21    | Individual images                           | href, title, width, height, credit |
  | internal       | 25    | Internal path reference (inside image href) | path                               |
  | square         | 8     | Square grid type (for map grids)            | size, offsetX, offsetY, scale      |
  | hexColsEven    | 2     | Hex grid type (for map grids)               | size, distance, units, offsetX/Y   |

主要关注包含文本内容的 type，如 section、entries、inset、insetReadaloud、list、item。

## ref tag

部分文本中包含 `@` 引用标签，用于链接到其他模组元素或提供格式化。 以下是支持的 ref 标签及其用途：

| Tag         | Purpose                              | Example Format                               |
  |-------------|--------------------------------------|----------------------------------------------|
  | @creature   | Monster/NPC reference                | {@creature Runara|DoSI}                      |
  | @spell      | Spell reference                      | {@spell flaming sphere}                      |
  | @item       | Magic item reference                 | {@item artisan's tools|PHB}                  |
  | @adventure  | Adventure section link               | {@adventure Chapter 1|DoSI|1}                |
  | @area       | Location/map area link               | {@area Dragon's Rest Locations|021|x}        |
  | @book       | Rulebook reference                   | {@book Basic Rules|PHB}                      |
  | @skill      | Skill check reference                | {@skill Perception}                          |
  | @condition  | Condition reference                  | {@condition incapacitated}                   |
  | @dc         | Difficulty class                     | {@dc 15}                                     |
  | @dice       | Dice notation                        | {@dice d4}                                   |
  | @action     | Action reference                     | {@action Dash}                               |
  | @background | Background reference                 | {@background Soldier||soldier background}    |
  | @sense      | Sense reference (darkvision, etc.)   | {@sense darkvision 60 ft.}                   |
  | @quickref   | Quick reference lookup               | {@quickref Some Reference}                   |
  | @b          | Bold text (formatting)               | {@b storyteller}                             |
  | @i          | Italic text (formatting)             | {@i Dragons of Stormwreck Isle}              |
  | @note       | Note/warning box                     | {@note This section was...}                  |
  | @5etoolsImg | Pre-generated character sheet images | {@5etoolsImg Hill Dwarf Cleric|pdf/DoSI/...} |

引用标签的格式通常为 `{@tag content|optional parameters}`，解析时需根据标签类型处理内容和参数。每个 section, entries, inset, insetReadaloud 都具有唯一的 id 字段，便于 {@area...} 和 {@adventure...} 引用和链接。同时，它们也有 name, page 和 entries 字段。

```
  Reference Tag Formats
  | Tag               | Format | Purpose |
  |-------------------|--------|---------|
  | `{@area NAME      | ID     | x}`     |
  | `{@adventure NAME | SOURCE | PAGE}`  |
```

以 `A5: Temple of Bahamut` 的引用为例，该小节定义了一个地点

```
{
    "type": "entries",
    "name": "A5: Temple of Bahamut",
    "page": 11,
    "id": "02c",                    // <-- Unique ID
    "entries": [
      {
        "type": "insetReadaloud",
        "id": "02d",                // <-- Child has its own ID
        "entries": [...]
      },
      "Description text..."
    ]
  }
```

并能通过 `{@area area A5|02c|x}` 进行引用，前者为网页渲染的文本，中间为唯一 ID。

{@creature}, {@spell}, {@item} 等标签引用的实体在其他文件中定义。

## json 结构

```
  data (root array)
  └── section (31 top-level sections)
      ├── entries[] (mixed content)
      │   ├── string (plain text paragraphs)
      │   ├── section (nested sections)
      │   ├── entries (named subsections)
      │   ├── inset (sidebar content)
      │   ├── insetReadaloud (boxed read-aloud text)
      │   ├── list
      │   │   └── items[] → item
      │   │       ├── name (optional)
      │   │       ├── entry (string)
      │   │       └── entries (array, for nested content)
      │   ├── table
      │   │   ├── rows (array of arrays)
      │   │   ├── colLabels (optional)
      │   │   └── caption (optional)
      │   ├── gallery
      │   │   └── images[]
      │   │       └── image
      │   │           ├── href {type: "internal", path: "..."}
      │   │           ├── title
      │   │           ├── imageType (map, mapPlayer, etc.)
      │   │           └── grid (optional) → square or hexColsEven
      │   └── image (standalone)
      └── id, name, page (metadata)
```

