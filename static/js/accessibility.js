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
                const focusTarget = closeButtons[0] || dialog;
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
            if (event.shiftKey && document.activeElement === first) {
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

    function enhancePlotAccessibility(root) {
        const scope = root || document;
        scope.querySelectorAll('.plot-container, .report-plot-placeholder').forEach(function(container) {
            const plotDiv = container.querySelector('.plotly-graph-div, .report-plotly-div');
            if (!plotDiv || plotDiv.dataset.accessibilityEnhanced === 'true') {
                return;
            }

            const containerId = container.id || ('plot-container-' + Math.random().toString(36).slice(2));
            container.id = containerId;
            const summaryId = containerId + '-accessibility-summary';
            let summary = document.getElementById(summaryId);
            const title = plotTitle(plotDiv, container);

            if (!summary) {
                summary = document.createElement('p');
                summary.id = summaryId;
                summary.className = 'sr-only plot-accessibility-summary';
                summary.textContent = title + '. Interactive chart. Use the Plotly toolbar to explore the chart. A data table containing the plotted values is available after the chart.';
                container.insertAdjacentElement('beforebegin', summary);
            }

            plotDiv.setAttribute('role', 'group');
            plotDiv.setAttribute('aria-roledescription', 'interactive chart');
            plotDiv.setAttribute('aria-label', title);
            plotDiv.setAttribute('aria-describedby', summaryId);

            const controlsContainer = container.nextElementSibling;
            const dataButton = controlsContainer && controlsContainer.classList.contains('custom-button-container')
                ? controlsContainer.querySelector('.show-data-btn')
                : null;
            if (dataButton) {
                const detailsId = dataButton.getAttribute('aria-controls');
                if (detailsId) {
                    plotDiv.setAttribute('aria-details', detailsId);
                }
                dataButton.dataset.plotTitle = title;
                dataButton.setAttribute('aria-label', 'Show data table for ' + title);
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
                    firstHeader.setAttribute('scope', firstHeader.hasAttribute('colspan') ? 'colgroup' : 'row');
                }
            });

            if (!table.querySelector('caption') && !table.hasAttribute('aria-label') && !table.hasAttribute('aria-labelledby')) {
                const caption = document.createElement('caption');
                caption.className = 'sr-only';
                caption.textContent = nearestHeadingText(table) || ('Data table ' + (index + 1));
                table.insertBefore(caption, table.firstChild);
            }
        });
    }

    function initializeDataTableControls(root) {
        const scope = root || document;
        scope.querySelectorAll('.show-data-btn').forEach(function(button, index) {
            const target = button.dataset.target;
            const container = target ? document.getElementById(target + '-container') : null;
            if (!container) {
                return;
            }
            if (!button.id) {
                button.id = 'show-data-table-' + (index + 1);
            }
            button.type = 'button';
            button.setAttribute('aria-controls', container.id);
            button.setAttribute('aria-expanded', container.hidden || getComputedStyle(container).display === 'none' ? 'false' : 'true');
            container.setAttribute('role', 'region');
            container.setAttribute('aria-labelledby', button.id);
            if (getComputedStyle(container).display === 'none') {
                container.hidden = true;
            }
        });
    }

    global.IVIVCAccessibility = {
        enhancePlotAccessibility: enhancePlotAccessibility,
        enhanceTableSemantics: enhanceTableSemantics,
        initializeDataTableControls: initializeDataTableControls,
        initializeHelpModal: initializeHelpModal,
        setLiveMessage: setLiveMessage
    };
})(window);
