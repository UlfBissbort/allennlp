"""
This is a tiny webapp for generating configuration stubs for your models.
It's very hacky and very experimental, so don't rely on it for anything important.

```
python -m allennlp.service.config_explorer
```

will launch the app on `localhost:8123` (you can specify a different port if you like).

It can also incorporate your own classes if you use the `include_package` flag:

```
python -m allennlp.service.config_explorer \
    --include-package my_library
```
"""
# pylint: disable=too-many-return-statements
from typing import Sequence
import argparse
import logging
import sys

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from allennlp.common.configuration import configure, Config
from allennlp.common.util import import_submodules
from allennlp.service.server_flask import ServerError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_app(include_packages: Sequence[str] = ()) -> Flask:
    """
    Creates a Flask app that serves up a simple configuration wizard.
    """
    # Load modules
    for package_name in include_packages:
        import_submodules(package_name)

    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response:  # pylint: disable=unused-variable
        return Response(response=_HTML, status=200)

    @app.route('/api/')
    def api() -> Response:  # pylint: disable=unused-variable
        class_name = request.args.get('class', '')

        config = configure(class_name)

        if isinstance(config, Config):
            return jsonify({
                    "className": class_name,
                    "configItems": config.to_json()
            })
        else:
            return jsonify({
                    "className": class_name,
                    "choices": config
            })

    return app


def main(args):
    parser = argparse.ArgumentParser(description='Serve up a simple configuration wizard')

    parser.add_argument('--port', type=int, default=8123, help='port to serve the wizard on')

    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')

    args = parser.parse_args(args)

    app = make_app(args.include_package)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print(f"Model loaded, serving demo on port {args.port}")
    http_server.serve_forever()

_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>AllenNLP Configuration Wizard (alpha)</title>
    <style>
        div {
            display: table;
        }

        * {
            font-family: sans-serif;
        }

        h1,
        h2 {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            font-weight: 300
        }

        h2 {
            font-size: 2em;
        }

        .optional {
            color: gray;
        }

        .required {
            color: black;
        }

        span.name {
            font-weight: bold;
            margin: 5px;
        }

        .required.incomplete > span.name {
            background-color: lightcoral;
        }

        .annotation {
            font-size: 90%;
            margin-left: 10px;
            margin-right: 10px;
            color: #2085bc;
        }

        .prefix {
            font-size: 75%;
            margin-left: 10px;
        }


        .default-value {
            color: #979a9d;
            font-size: 90%;
        }

        div#rendered-json {
            margin: 10px;
            border: 1px solid black;
            font-family: monospace;
            white-space: pre;
        }

        .config-item {
            margin-top: 2px;
        }

        .tippy-content {
            color: white;
        }
    </style>
  </head>
  <body>
    <h2>AllenNLP Configuration Wizard (alpha)</h2>

    <div id="app"></div>

    <script crossorigin src="https://unpkg.com/react@16/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@16/umd/react-dom.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/6.26.0/babel.js"></script>
    <script src="https://unpkg.com/tippy.js@2.5.2/dist/tippy.all.min.js"></script>
    <script type="text/babel">

// A "config" is empty if it's undefined, if it has zero length
// (i.e. is an empty string or list), or if it's an empty object.
function isEmpty(config) {
    return (
        config === undefined ||
        config.length == 0 || (config.constructor === Object && Object.keys(config).length == 0)
    )
}

function renderAnnotation(annotation) {
    if (!annotation) {
        return ""
    }

    const [origin, ...args] = annotation

    if (args.length == 0) {
        return origin
    } else {

        return origin + "[" + args.map(renderAnnotation).join(",") + "]"
    }
}

function renderDefaultValue(defaultValue) {
    if (defaultValue === undefined) {
        return null
    } else {
        return "(default: " + defaultValue + ")"
    }
}

// Sometimes we don't have type annotations
// (e.g. for torch classes)
// and we have to best guess how to serialize an input.
function bestGuess(x) {
    const asNumber = +x

    if (x === "true") {
        return true
    } else if (x === "false") {
        return false
    } else if (!isNaN(asNumber)) {
        return asNumber
    } else {
        // Assume string
        return x
    }
}

// Represent some config item as JSON.
function configToJson(config, annotation) {
    const origin = annotation[0]

    if (origin === "?") {
        // No type annotation, so do a best guess
        return bestGuess(config)
    }

    else if (origin === "str") {
        // strings can stay as is
        return config

    } else if (origin === "bool") {
        // return only valid booleans
        if (config === "true") {
            return true
        } else if (config === "false") {
            return false
        } else {
            return undefined
        }

    } else if (origin === "int" || origin === "float") {
        // return only valid numbers
        if (isNaN(config)) {
            return undefined
        } else {
            return +config
        }

    } else if (origin === "List" || origin === "Sequence") {
        // recurse on List items and throw away undefined ones
        // (e.g. missing items in a list)
        return config
            .map((item) => configToJson(item.value, annotation[1]))
            .filter((item) => item)

    } else if (origin === "Dict") {
        // recurse on Dict items
        let blob = {}
        config.forEach((item) => blob[item.key] = configToJson(item.value, annotation[2]))
        return blob
    } else {
        // otherwise just return an empty dictionary
        return {}
    }
}

class App extends React.Component {
    constructor() {
        super()

        // config is the state used for rendering the page
        // json is what gets rendered out
        // this is not a good design
        this.state = {config: {}, json: {}, renderedJson: ""}

        this.generate = this.generate.bind(this)
        this.setJson = this.setJson.bind(this)
    }

    generate() {
        // Generate the configuration JSON
        this.setState({renderedJson: JSON.stringify(this.state.json, null, 4)})
    }

    setJson(path, childConfig, annotation) {
        // add the specified childConfig with the specified annotation
        // to the master json object at the given path
        let {json} = this.state

        path.forEach((name, idx) => {
            // walk the first n-1 steps of the path

            if (idx < path.length - 1) {
                json = json[name]
            } else {
                // and add the jsonized item if it isn't empty
                // delete it if it is
                if (!isEmpty(childConfig) && name !== undefined) {
                    json[name] = configToJson(childConfig, annotation)
                } else {
                    delete json[name]
                }
            }
        })
    }

    componentDidMount() {
        // Fetch the top level configuration
        fetch('/api/')
            .then(res => res.json())
            .then(config => this.setState({config}))
    }

    // If you click anywhere in the rendered JSON config, it selects the whole thing.
    selectRenderedJson() {
        window.getSelection().selectAllChildren(document.getElementById('rendered-json'))
    }

    render() {
        return (
            <div class="wizard">
                <Configuration path={[]} config={this.state.config} setJson={this.setJson} optional={false} setCompleted={() => {}}/>
                <div>
                    <button id="generate" onClick={this.generate}>GENERATE</button>
                </div>
                <div id="rendered-json" onClick={this.selectRenderedJson}>
                    {this.state.renderedJson}
                </div>
            </div>
        )
    }
}

class Configuration extends React.Component {
    // Component representing a Config as returned by an API call
    constructor(props) {
        super(props)

        let completed = []

        const items = (props.config && props.config.configItems) || []
        items.forEach((item) => {
            if (item.defaultValue === undefined && !item.type) {
                // Required, so use false
                completed.push(false)
            } else {
                // Optional, so use null
                completed.push(null)
            }
        })

        this.setCompleted = (idx) => ((c) => {
            completed = this.state.completed.slice()
            if (completed[idx] !== null) {
                completed[idx] = c
                const allComplete = completed.filter((x) => x !== false).length > 0

                props.setCompleted(allComplete)
                this.setState({completed, allComplete})
            }
        })

        const allComplete = completed.filter((x) => x !== false).length > 0
        if (allComplete) {
            props.setCompleted(allComplete)
        }

        this.state = {
            completed: completed,
            allComplete: allComplete
        }
    }

    componentDidMount() {
        // I hope this is idempotent
        tippy('.tooltip')
    }

    render() {
        // "config" contains the items making up JSON blob as returned by the API

        const {config, path, setJson} = this.props
        const {configItems} = config

        let className = "configuration"
        if (this.props.optional) {
            className += " optional"
        } else {
            className += " required"
        }

        if (!this.state.allComplete) {
            className += " incomplete"
        }

        if (configItems !== undefined) {
            return (
                <div class="configuration">
                    <ul>
                        {configItems.map((configItem) => <ConfigItem setCompleted={this.setCompleted} path={path} item={configItem} setJson={setJson}/>)}
                    </ul>
                </div>
            )
        } else {
            return null
        }
    }
}

class DictOrList extends React.Component {
    constructor(props) {
        super(props)

        this.state = {items: []}
        this.remove = this.remove.bind(this)
        this.update = this.update.bind(this)
        this.add = this.add.bind(this)
        this.reportItemCount = this.reportItemCount.bind(this)
    }

    reportItemCount() {
        const itemCount = this.state.items.filter((item) => {
            return item.key || item.value
        }).length

        this.props.setCompleted(itemCount)
    }

    remove(idx) {
        return ((i) => (() => {
            const {items} = this.state
            const beforeItems = items.slice(0, i)
            const afterItems = items.slice(i + 1)
            const newItems = beforeItems.concat(afterItems)
            this.props.setJson(this.props.path, newItems, this.props.annotation)
            this.setState({items: newItems}, this.reportItemCount)
        }))(idx)
    }

    update(key, idx) {
        return ((i) => ((e) => {
            const value = e.target.value
            const newItems = this.state.items.slice()
            newItems[i][key] = value
            this.props.setJson(this.props.path, newItems, this.props.annotation)
            this.setState({items: newItems}, this.reportItemCount)
        }))(idx)
    }

    add() {
        this.setState({items: this.state.items.concat([{}])})
    }

    componentDidMount() {
        // I hope this is idempotent
        tippy('.tooltip')
    }

    render() {
        const {annotation, value, defaultValue, setJson} = this.props
        const isDict = annotation[0] == "Dict"
        let keyType, valueType
        if (isDict) {
            keyType = annotation[1]
            valueType = annotation[2]
        } else {
            valueType = annotation[1]
        }

        // This is probably a bad assumption
        const configurable = valueType[0].startsWith('allennlp.') || valueType[0].startsWith('torch.')
        const optional = false

        const renderedItems = this.state.items.map((item, idx) => {
            const {key, value} = item

            let path
            if (annotation[0] == "Dict") {
                path = this.props.path.concat(key)
            } else {
                path = this.props.path.concat(idx)
            }
            const optional = false

            const configureButtonDisabled = isDict && !key

            const valueInput = renderValue(configurable,
                                           optional,
                                           path,
                                           valueType, // annotation,
                                           setJson,
                                           this.reportItemCount,
                                           defaultValue,
                                           configureButtonDisabled) //


            return (
                <span>
                    <div>
                        <button tabIndex="-1" class="remove-button" onClick={this.remove(idx)}>X</button>
                        {keyType ? <input type="text" value={key} onChange={this.update("key", idx)}/> : null}
                        {valueInput}
                    </div>
                </span>
            )
        })

        return (
            <div>
                {renderedItems}
                <div><button tabIndex="-1" class="add-button" onClick={this.add}>+</button></div>
            </div>
        )
    }

}

function renderValue(configurable, optional, path, annotation, setJson, setCompleted, defaultValue, buttonDisabled) {
    if (configurable) {
        return <Configurator optional={optional} path={path} annotation={annotation} setJson={setJson} setCompleted={setCompleted} defaultValue={defaultValue} buttonDisabled={buttonDisabled}/>
    } else if (annotation[0] === "Dict" || annotation[0] === "List" || annotation[0] === "Sequence" || (annotation[0] == "Tuple" && annotation[2] == "...")) {
        return <DictOrList path={path} setJson={setJson} annotation={annotation} setCompleted={setCompleted}/>
    } else {
        return (
            <span>
                <TextInput path={path} setJson={setJson} annotation={annotation} setCompleted={setCompleted}/>
                <span class="annotation">{renderAnnotation(annotation)}</span>
                <span class="default-value">{renderDefaultValue(defaultValue)}</span>
            </span>
        )
    }
}

class ConfigItem extends React.Component {
    constructor(props) {
        super(props)
        const {path, item, setJson} = props
        const {name, annotation, configurable, defaultValue} = item
        this.optional = defaultValue !== undefined

        this.state = {completed: false, configurable: configurable, configured: false}
        this.setCompleted = this.setCompleted.bind(this)
    }

    setCompleted(completed) {
        this.props.setCompleted(completed)
        this.setState({completed})
    }

    componentDidMount() {
        // I hope this is idempotent
        tippy('.tooltip')
    }

    render() {
        const {props, state, optional} = this
        const {item, setJson, path} = props
        const {name, annotation, configurable, defaultValue, comment, subconfig} = item
        const {completed} = state

        const newPath = path.concat(name)
        const newPathStr = newPath.join('-')

        const indent = newPath.length
        const marginLeft = 30 * newPath.length

        // Short circuit for type
        if (item.type) {
            setJson(newPath, item.type, ["str"])
            return <div key={newPathStr} marginLeft={marginLeft} class="config-item"><span class="name">type</span>: {item.type}</div>
        }

        const renderedValue = renderValue(configurable, optional, newPath, annotation, setJson, this.setCompleted, defaultValue)

        let className = "config-item"
        if (optional) {
            className += " optional"
        } else {
            className += " required"
        }
        if (!completed) {
            className += " incomplete"
        }

        const tooltip = comment ? <button tabIndex="-1" id={newPathStr + '-tooltip'} key={newPathStr + '-tooltip'} class="tooltip" title={comment}>?</button> : null

        return (
            <div key={newPathStr} marginLeft={marginLeft} className={className}>
                <span class="name">{name}</span>
                {tooltip}
                <span>:</span>
                {renderedValue}
            </div>
        )
    }
}

class TextInput extends React.Component {
    constructor(props) {
        super(props)

        this.state = {value: ''}

        this.valueChange = this.valueChange.bind(this)
    }

    valueChange(e) {
        const {path, annotation, setCompleted} = this.props
        const value = e.target.value

        this.props.setCompleted(value)
        this.props.setJson(path, value, annotation)
        this.setState({value})
    }

    render() {
        return (<input type="text" value={this.state.value} onChange={this.valueChange}/>)
    }
}

function sharedPrefix(s1, s2) {
    const length = Math.min(s1.length, s2.length)
    for (let i = 0; i < length; i++) {
        if (s1[i] !== s2[i]) {
            return s1.slice(0, i)
        }
    }

    return s1.slice(0, length)
}

function commonPrefix(strings) {
    if (strings.length == 1) {
        // Special case, return up to last .
        const string = strings[0]
        const idx = string.lastIndexOf(".")
        return string.slice(0, idx + 1)
    } else {
        return strings.reduce(sharedPrefix)
    }
}

class Configurator extends React.Component {
    constructor(props) {
        super(props)

        const {path, annotation, optional, setJson, setCompleted, buttonDisabled} = props

        this.state = {
            childConfig: null,
            subclasses: null,
            value: null,
            choice: null,
            completed: false
        }

        this.subconfigure = this.subconfigure.bind(this)
        this.remove = this.remove.bind(this)
        this.valueChange = this.valueChange.bind(this)
        this.select = this.select.bind(this)
        this.updateJson = this.updateJson.bind(this)
        this.setValue = this.setValue.bind(this)
        this.setCompleted = this.setCompleted.bind(this)
    }

    updateJson(childConfig) {
        this.props.setJson(this.props.path, childConfig, this.props.annotation)
    }

    remove() {
        this.updateJson(undefined)
        this.props.setCompleted(false)
        this.setState({childConfig: null, subclasses: null, value: null, choice: ''})
    }

    setCompleted(completed) {
        this.props.setCompleted(completed)
        this.setState({completed: completed})
    }

    setValue(value) {
        this.setState({value: value})
    }

    valueChange(e) {
        const value = e.target.value
        this.updateJson(value)
        this.setState({value: value})
    }

    select(e) {
        const subclass = e.target.value

        if (subclass) {
            this.updateJson(subclass)
            fetch('/api/?class=' + subclass)
                .then(res => res.json())
                .then(config => {
                    this.setState({
                        childConfig: config,
                        choice: subclass
                    })
                })
        } else {
            this.remove()
        }
    }

    subconfigure(e) {
        const [subname] = this.props.annotation;

        fetch("/api/?class=" + subname)
            .then(res => res.json())
            .then(config => {
                if (config.configItems) {
                    this.updateJson(config)
                    this.setState({childConfig: config})
                } else {
                    this.setState({subclasses: config.choices})
                }
            })
    }

    render() {
        const {subclasses, childConfig, value, choice} = this.state
        const {buttonDisabled} = this.props
        const hasData = childConfig || subclasses || value

        let input = null
        if (subclasses) {
            const prefix = commonPrefix(subclasses)

            input = (
                <span>
                    <span class="prefix">{prefix}</span>
                    <select value={choice} onChange={this.select}>
                        {choice ? null : (<option value=""></option>)}
                        {subclasses.map((subclass) => <option value={subclass}>{subclass.slice(prefix.length)}</option>)}
                    </select>
                </span>
            )
        } else if (!hasData) {
            input = (<button tabIndex="-1" disabled={buttonDisabled} class="subconfigure" onClick={this.subconfigure}>CONFIGURE</button>)
        }

        const remove = hasData ? (<button tabIndex="-1" class="remove-button" onClick={this.remove}>X</button>) : null

        let child = null
        if (childConfig) {
            child = (<Configuration path={this.props.path} config={childConfig} setJson={this.props.setJson} setCompleted={this.setCompleted}/>)
        }

        let renderedAnnotation = null
        let renderedDefaultValue = null
        if (!hasData) {
            renderedAnnotation = (<span class="annotation">{renderAnnotation(this.props.annotation)}</span>)
            renderedDefaultValue = (<span class="default-value">{renderDefaultValue(this.props.defaultValue)}</span>)
        }

        return (
            <span>
                {input}
                {remove}
                {renderedAnnotation}
                {renderedDefaultValue}
                {child}

            </span>
        )
    }
}

ReactDOM.render(<App />, document.getElementById("app"))
    </script>

  </body>
</html>"""


if __name__ == "__main__":
    main(sys.argv[1:])
