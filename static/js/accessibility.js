(function(global) {
    'use strict';

    const focusableSelector = [
        'a[href]',
        'area[href]',
        'button:not([disabled])',
        'input:not([disabled]):not([type="hidden"])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        'iframe',
        'object',
        'embed',
        '[contenteditable="true"]',
        '[tabindex]:not([tabindex="-1"])'
    ].join(',');

    function setLiveMessage(elementOrId, message) {
        const element = typeof elementOrId === 'string'
            ? document.getElementById(elementOrId)
            : elementOrId;
        if (!element) {
            return;
        }
        element.textContent = '';
        window.setTimeout(function() {
            element.textContent = message || '';
        }, 20);
    }

    function stripHtml(value) {
        const container = document.createElement('div');
        container.innerHTML = value || '';
        return (container.textContent || '').replace(/\s+/g, ' ').trim();
    }

    function contextualTitle(trigger) {
        const label = trigger.getAttribute('aria-label') || '';
        return label.replace(/^Help (?:for|about)\s+/i, '').trim() || 'Help';
    }

    function initializeHelpModal(contentMap) {
        const modal = document.getElementById('help-modal');
        if (!modal) {
            return;
        }

        const dialog = modal.querySelector('.modal-content');
        const modalText = document.getElementById('modal-text');
        const modalTitle = document.getElementById('modal-title');
        const closeButtons = Array.from(modal.querySelectorAll('.close, .modal-close-btn'));
        let lastTrigger = null;
        let inertedElements = [];

        function setBackgroundInert(isInert) {
            if (isInert) {
                inertedElements = Array.from(document.body.children).filter(function(element) {
                    return element !== modal && element instanceof HTMLElement && !element.hasAttribute('inert');
                });
                inertedElements.forEach(function(element) {
                    element.inert = true;
                });
            } else {
                inertedElements.forEach(function(element) {
                    element.inert = false;
                });
                inertedElements = [];
            }
        }

        function getFocusableElements() {
            return Array.from(dialog.querySelectorAll(focusableSelector)).filter(function(element) {
                return !element.hidden && element.offsetParent !== null;
            });
        }

        function openModal(trigger) {
            const modalId = trigger.dataset.modal;
            const contentId = contentMap[modalId];
            const source = contentId ? document.getElementById(contentId) : null;

            lastTrigger = trigger;
            modalText.innerHTML = source
                ? source.innerHTML
                : '<h4>Help</h4><p>Help information for this section is not available.</p>';
            modalTitle.textContent = contextualTitle(trigger);
            modal.hidden = false;
            modal.setAttribute('aria-hidden', 'false');
            document.body.classList.add('modal-open');
            setBackgroundInert(true);

            window.requestAnimationFrame(function() {
                const focusTarget = modalTitle || closeButtons[0] || dialog;
                focusTarget.focus();
            });
        }

        function closeModal() {
            if (modal.hidden) {
                return;
            }
            modal.hidden = true;
            modal.setAttribute('aria-hidden', 'true');
            document.body.classList.remove('modal-open');
            setBackgroundInert(false);
            if (lastTrigger && document.contains(lastTrigger)) {
                lastTrigger.focus();
            }
            lastTrigger = null;
        }

        document.querySelectorAll('[data-modal]').forEach(function(trigger) {
            trigger.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation();
                openModal(trigger);
            });
        });

        closeButtons.forEach(function(button) {
            button.addEventListener('click', closeModal);
        });

        modal.addEventListener('click', function(event) {
            if (event.target === modal) {
                closeModal();
            }
        });

        dialog.addEventListener('click', function(event) {
            event.stopPropagation();
        });

        document.addEventListener('keydown', function(event) {
            if (modal.hidden) {
                return;
            }
            if (event.key === 'Escape') {
                event.preventDefault();
                closeModal();
                return;
            }
            if (event.key !== 'Tab') {
                return;
            }

            const focusable = getFocusableElements();
            if (focusable.length === 0) {
                event.preventDefault();
                dialog.focus();
                return;
            }

            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && (document.activeElement === first || document.activeElement === modalTitle || !dialog.contains(document.activeElement))) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        });
    }

    function nearestHeadingText(element) {
        let current = element;
        while (current && current !== document.body) {
            let sibling = current.previousElementSibling;
            while (sibling) {
                if (/^H[1-6]$/.test(sibling.tagName)) {
                    return sibling.textContent.replace(/\s+/g, ' ').trim();
                }
                const nestedHeading = sibling.querySelector && sibling.querySelector('h1, h2, h3, h4, h5, h6');
                if (nestedHeading) {
                    return nestedHeading.textContent.replace(/\s+/g, ' ').trim();
                }
                sibling = sibling.previousElementSibling;
            }
            current = current.parentElement;
        }
        return '';
    }

    function plotTitle(plotDiv, container) {
        const layoutTitle = plotDiv && plotDiv.layout && plotDiv.layout.title
            ? plotDiv.layout.title.text || plotDiv.layout.title
            : '';
        return stripHtml(layoutTitle) || nearestHeadingText(container) || 'Interactive data plot';
    }

    function axisTitle(plotDiv, axisName, fallback) {
        const axis = plotDiv && plotDiv.layout ? plotDiv.layout[axisName] : null;
        const title = axis && axis.title ? axis.title.text || axis.title : '';
        return stripHtml(title) || fallback;
    }

    function uniqueTraceNames(plotDiv) {
        const names = [];
        (plotDiv.data || []).forEach(function(trace) {
            const name = stripHtml(trace.name || '');
            if (!name || /upper bound/i.test(name) || names.indexOf(name) !== -1) {
                return;
            }
            names.push(name);
        });
        return names;
    }

    function hasErrorBars(plotDiv) {
        return (plotDiv.data || []).some(function(trace) {
            return Boolean(
                (trace.error_x && trace.error_x.visible !== false && trace.error_x.array) ||
                (trace.error_y && trace.error_y.visible !== false && trace.error_y.array)
            );
        });
    }

    function hasPredictionBand(plotDiv) {
        return (plotDiv.data || []).some(function(trace) {
            return /prediction band/i.test(trace.name || '') || trace.fill === 'tonexty';
        });
    }

    function associatedUncertaintyText(container) {
        let sibling = container.previousElementSibling;
        while (sibling) {
            if (sibling.classList && sibling.classList.contains('uncertainty-plot-alert')) {
                return sibling.textContent.replace(/\s+/g, ' ').trim();
            }
            if (/^H[1-6]$/.test(sibling.tagName) || (sibling.classList && sibling.classList.contains('plot-container'))) {
                break;
            }
            sibling = sibling.previousElementSibling;
        }
        return '';
    }

    function plotTypeDescription(container, plotDiv, title) {
        const id = (container.id || '').toLowerCase();
        const titleLower = title.toLowerCase();
        const traces = uniqueTraceNames(plotDiv);
        const traceText = traces.length
            ? ' Series shown: ' + traces.join('; ') + '.'
            : '';
        const xTitle = axisTitle(plotDiv, 'xaxis', 'x-axis value');
        const yTitle = axisTitle(plotDiv, 'yaxis', 'y-axis value');
        let explanation = '';

        if (id.indexOf('residual-qq') !== -1 || titleLower.indexOf('q-q') !== -1) {
            explanation = ' The diagonal reference line shows the pattern expected for normally distributed standardized residuals. Points farther from the line indicate larger departures from that pattern.';
        } else if (id.indexOf('residual') !== -1 || titleLower.indexOf('residual') !== -1) {
            explanation = ' The dashed horizontal line marks zero residual. When present, the shaded inner band and dotted outer lines provide common visual reference ranges for comparing residual magnitude; they are not acceptance limits or confidence intervals.';
        } else if (id.indexOf('processing-raw-data') !== -1) {
            explanation = ' Markers show the uploaded in vitro and in vivo observations before preprocessing.';
        } else if (id.indexOf('interpolation') !== -1) {
            explanation = ' Marker symbols distinguish measured observations from values estimated by interpolation.';
        } else if (id.indexOf('plot-fit-approach3') !== -1) {
            explanation = ' The plot shows the direct interpolation-based value or time ratio used for mapping; no parametric fitted curve or model-based uncertainty band is shown.';
        } else if (id.indexOf('plot-fit-') !== -1 || titleLower.indexOf(' fit') !== -1) {
            explanation = ' Markers show observations and lines show the fitted model relationship.';
            if (hasPredictionBand(plotDiv)) {
                explanation += ' Shaded areas show approximate 95% prediction bands.';
            }
        } else if (id.indexOf('prediction') !== -1 || titleLower.indexOf('prediction') !== -1) {
            explanation = ' Marker symbols distinguish fitting observations, prediction inputs, and predicted in vivo values or times. Dotted connecting lines, when present, show the corresponding mapping between input and prediction points.';
        }

        if (hasErrorBars(plotDiv)) {
            explanation += ' Error bars show the calculated uncertainty around predicted values or times.';
        }

        const warning = associatedUncertaintyText(container);
        if (warning) {
            explanation += ' Plot warning: ' + warning + '.';
        }

        return title + '. Horizontal axis: ' + xTitle + '. Vertical axis: ' + yTitle + '.' + traceText + explanation + ' Use the Plotly toolbar to inspect or rescale the chart. The controls beneath the plot can copy or download the plotted series and any available error-bar bounds.';
    }

    function enhancePlotAccessibility(root) {
        const scope = root || document;
        scope.querySelectorAll('.plot-container, .report-plot-placeholder').forEach(function(container) {
            const plotDiv = container.querySelector('.plotly-graph-div, .report-plotly-div');
            if (!plotDiv) {
                return;
            }

            const containerId = container.id || ('plot-container-' + Math.random().toString(36).slice(2));
            container.id = containerId;
            const summaryId = containerId + '-accessibility-summary';
            let summary = document.getElementById(summaryId);
            const title = plotTitle(plotDiv, container);
            const description = plotTypeDescription(container, plotDiv, title);

            if (!summary) {
                summary = document.createElement('p');
                summary.id = summaryId;
                summary.className = 'sr-only plot-accessibility-summary';
                container.insertAdjacentElement('beforebegin', summary);
            }
            summary.textContent = description;

            plotDiv.setAttribute('role', 'group');
            plotDiv.setAttribute('aria-roledescription', 'interactive chart');
            plotDiv.setAttribute('aria-label', title);
            plotDiv.setAttribute('aria-describedby', summaryId);

            const controlsContainer = container.nextElementSibling;
            const downloadGroup = controlsContainer && controlsContainer.classList.contains('plot-data-downloads')
                ? controlsContainer
                : null;
            if (downloadGroup) {
                if (!downloadGroup.id) {
                    downloadGroup.id = containerId + '-data-downloads';
                }
                downloadGroup.dataset.plotTitle = title;
                downloadGroup.setAttribute('aria-label', 'Plot data downloads for ' + title);
                plotDiv.setAttribute('aria-details', downloadGroup.id);
            }
            plotDiv.dataset.accessibilityEnhanced = 'true';
        });
    }

    function enhanceTableSemantics(root) {
        const scope = root || document;
        scope.querySelectorAll('table').forEach(function(table, index) {
            table.querySelectorAll('thead th').forEach(function(header) {
                if (!header.hasAttribute('scope')) {
                    header.setAttribute('scope', 'col');
                }
            });
            table.querySelectorAll('tbody tr').forEach(function(row) {
                const firstHeader = row.querySelector(':scope > th');
                if (firstHeader && !firstHeader.hasAttribute('scope')) {
                    firstHeader.setAttribute(
                        'scope',
                        row.classList.contains('settings-section-row') ? 'rowgroup' : 'row'
                    );
                }
            });

            if (!table.querySelector('caption') && !table.hasAttribute('aria-label') && !table.hasAttribute('aria-labelledby')) {
                const caption = document.createElement('caption');
                caption.className = 'sr-only';
                caption.textContent = nearestHeadingText(table) || ('Data table ' + (index + 1));
                table.insertBefore(caption, table.firstChild);
            }

            if (!table.closest('.data-table-container, .table-scroll-region, .dt-container, .dataTables_wrapper')) {
                const caption = table.querySelector('caption');
                const region = document.createElement('div');
                const label = caption ? caption.textContent.replace(/\s+/g, ' ').trim() : ('Data table ' + (index + 1));
                region.className = 'table-scroll-region';
                region.dataset.tableRegionLabel = label;
                table.parentNode.insertBefore(region, table);
                region.appendChild(table);
            }
        });

        window.requestAnimationFrame(function() {
            updateTableScrollRegions(scope);
        });
    }

    function updateTableScrollRegion(region) {
        const label = region.dataset.tableRegionLabel || nearestHeadingText(region) || 'data table';
        const isVisible = region.offsetParent !== null;
        const hasHorizontalOverflow = isVisible && region.scrollWidth > region.clientWidth + 1;

        if (hasHorizontalOverflow) {
            region.setAttribute('role', 'region');
            region.setAttribute('aria-label', 'Scrollable table: ' + label);
            region.setAttribute('tabindex', '0');
            region.dataset.horizontallyScrollable = 'true';
        } else {
            region.removeAttribute('role');
            region.removeAttribute('aria-label');
            region.removeAttribute('tabindex');
            delete region.dataset.horizontallyScrollable;
        }
    }

    function updateTableScrollRegions(root) {
        const scope = root || document;
        const regions = [];
        if (scope instanceof Element && scope.classList.contains('table-scroll-region')) {
            regions.push(scope);
        }
        scope.querySelectorAll('.table-scroll-region').forEach(function(region) {
            regions.push(region);
        });
        regions.forEach(updateTableScrollRegion);
    }

    function initializeDataTableControls(root) {
        const scope = root || document;
        scope.querySelectorAll('.plot-data-downloads').forEach(function(group, index) {
            const target = group.dataset.tableTarget;
            const sourceContainer = target ? document.getElementById(target + '-container') : null;
            const title = group.dataset.plotTitle || 'plot';

            if (!group.id) {
                group.id = 'plot-data-downloads-' + (index + 1);
            }
            group.setAttribute('role', 'group');
            group.setAttribute('aria-label', 'Plot data downloads for ' + title);

            if (sourceContainer) {
                sourceContainer.hidden = true;
                sourceContainer.setAttribute('aria-hidden', 'true');
            }
        });
    }

    global.IVIVCAccessibility = {
        enhancePlotAccessibility: enhancePlotAccessibility,
        enhanceTableSemantics: enhanceTableSemantics,
        initializeDataTableControls: initializeDataTableControls,
        initializeHelpModal: initializeHelpModal,
        setLiveMessage: setLiveMessage,
        updateTableScrollRegions: updateTableScrollRegions
    };

    let resizeTimer = null;
    window.addEventListener('resize', function() {
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(function() {
            updateTableScrollRegions(document);
        }, 100);
    });
})(window);
